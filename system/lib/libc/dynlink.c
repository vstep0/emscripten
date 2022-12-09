/*
 * Copyright 2021 The Emscripten Authors.  All rights reserved.
 * Emscripten is available under two separate licenses, the MIT license and the
 * University of Illinois/NCSA Open Source License.  Both these licenses can be
 * found in the LICENSE file.
 *
 * Emscripten-specific version dlopen and associated functions.  Some code is
 * shared with musl's ldso/dynlink.c.
 */

#define _GNU_SOURCE
#include <assert.h>
#include <dlfcn.h>
#include <pthread.h>
#include <threads.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dynlink.h>

#include <emscripten/console.h>
#include <emscripten/threading.h>
#include <emscripten/proxying.h>

//#define DYLINK_DEBUG

struct async_data {
  em_dlopen_callback onsuccess;
  em_arg_callback_func onerror;
  void* user_data;
};
typedef void (*dlopen_callback_func)(struct dso*, void* user_data);

void _dlinit(struct dso* main_dso_handle);
void* _dlopen_js(struct dso* handle);
void* _dlsym_js(struct dso* handle, const char* symbol);
void _emscripten_dlopen_js(struct dso* handle,
                           dlopen_callback_func onsuccess,
                           dlopen_callback_func onerror,
                           void* user_data);
void __dl_vseterr(const char*, va_list);

static struct dso * _Atomic head, * _Atomic tail;
static thread_local struct dso* thread_local_tail;
static pthread_rwlock_t lock;
static pthread_mutex_t dlopen_mutex;

static void error(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  __dl_vseterr(fmt, ap);
  va_end(ap);
#ifdef DYLINK_DEBUG
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
#endif
}

int __dl_invalid_handle(void* h) {
  struct dso* p;
  for (p = head; p; p = p->next)
    if (h == p)
      return 0;
  error("Invalid library handle %p", (void*)h);
  return 1;
}

static void load_library_done(struct dso* p) {
#ifdef DYLINK_DEBUG
  _emscripten_errf("%p: load_library_done: dso=%p mem_addr=%p mem_size=%zu "
                   "table_addr=%p table_size=%zu",
                   pthread_self(),
                   p,
                   p->mem_addr,
                   p->mem_size,
                   p->table_addr,
                   p->table_size);
#endif

  // insert into linked list
  p->prev = tail;
  if (tail) {
    tail->next = p;
  }
  tail = p;
  thread_local_tail = p;

  if (!head) {
    head = p;
  }
}

static struct dso* load_library_start(const char* name, int flags) {
  if (!(flags & (RTLD_LAZY | RTLD_NOW))) {
    error("invalid mode for dlopen(): Either RTLD_LAZY or RTLD_NOW is required");
    return NULL;
  }

  struct dso* p;
  size_t alloc_size = sizeof *p + strlen(name) + 1;
  p = calloc(1, alloc_size);
  p->flags = flags;
  strcpy(p->name, name);

  return p;
}

// This function is called at the start of all entry points so that the dso
// list gets initialized on first use.
static void ensure_init() {
  if (head) {
    return;
  }
  // Initialize the dso list.  This happens on first run.
  pthread_rwlock_wrlock(&lock);
  if (!head) {
    // Flags are not important since the main module is already loaded.
    struct dso* p = load_library_start("__main__", RTLD_NOW|RTLD_GLOBAL);
    assert(p);
    _dlinit(p);
    load_library_done(p);
    assert(head);
  }
  pthread_rwlock_unlock(&lock);
}

#ifdef _REENTRANT
// When we are attempting the syncronize loaded libraries between threads we
// currently abort, rather than rejecting the promises.  We could reject the
// promises, and attempt to return an error from the original dlopen() but we
// would have to also unwind the state on all the threads that were able to load
// the module.
#define ABORT_ON_SYNC_FAILURE 1

// These functions are defined in JS.

// Asynchronous version of "sync_all_threads".  Called only on the main thread.
// Runs _emscripten_thread_sync_code on each of threads that are running at the
// time of the call.  Once this is done the callback is called with the given
// em_proxying_ctx.
void _emscripten_sync_all_threads_async(pthread_t calling_thread,
                                        void (*callback)(em_proxying_ctx*),
                                        em_proxying_ctx* ctx);

// Synchronous version of "sync_all_threads".  Called only on the main thread.
// Runs _emscripten_thread_sync_code on each of threads that are running at the
// time of the call.
void _emscripten_sync_all_threads();

static void sync_next(struct dso* dso, int promise_id);

static void sync_one_onsuccess(struct dso* dso, void* user_data) {
  int promise_id = (intptr_t)user_data;
  // Load the next dso in the list
  thread_local_tail = dso;
  sync_next(dso->next, promise_id);
}

static void sync_one_onerror(struct dso* dso, void* user_data) {
#if ABORT_ON_SYNC_FAILURE
  abort();
#else
  int promise_id = (intptr_t)user_data;
  pthread_rwlock_unlock(&lock);
  emscripten_promise_reject(promise_id);
#endif
}

// Called on the main thread to asynchronously "catch up" with all the DSOs
// that are currently loaded.
static void sync_next(struct dso* dso, int promise_id) {
  if (!dso) {
    // All dso loaded
    pthread_rwlock_unlock(&lock);
    emscripten_promise_resolve(promise_id, NULL);
    return;
  }

  _emscripten_dlopen_js(
    dso, sync_one_onsuccess, sync_one_onerror, (void*)promise_id);
}

void _emscripten_thread_sync_code_async(void* user_data, int promise_id) {
#ifdef DYLINK_DEBUG
  _emscripten_errf("_emscripten_thread_sync_code_async promise_id=%d", promise_id);
#endif
  ensure_init();
  // Unlock happens once all DSO have been loaded, or one of them fails
  // with sync_one_onerror.
  pthread_rwlock_rdlock(&lock);
  if (!thread_local_tail) {
    thread_local_tail = head;
  }
  sync_next(thread_local_tail->next, promise_id);
}

// Called on background threads to synchronously "catch up" with all the DSOs
// that are currently loaded.
bool _emscripten_thread_sync_code() {
  // Should only ever be called from a background thread.
  assert(!emscripten_is_main_runtime_thread());
  ensure_init();
  if (thread_local_tail == tail) {
#ifdef DYLINK_DEBUG
    _emscripten_errf("%p: _emscripten_thread_sync_code: already in sync", pthread_self());
#endif
    return true;
  }
  pthread_rwlock_rdlock(&lock);
  if (!thread_local_tail) {
    thread_local_tail = head;
  }
  while (thread_local_tail->next) {
    struct dso* p = thread_local_tail->next;
#ifdef DYLINK_DEBUG
    _emscripten_errf(
      "%p: _emscripten_thread_sync_code: %s mem_addr=%p mem_size=%zu "
      "table_addr=%p table_size=%zu",
      pthread_self(),
      p->name,
      p->mem_addr,
      p->mem_size,
      p->table_addr,
      p->table_size);
#endif
    void* success = _dlopen_js(p);
    if (!success) {
      pthread_rwlock_unlock(&lock);
      _emscripten_errf("%p: _emscripten_thread_sync_code failed: %s", pthread_self(), dlerror());
      return false;
    }
    thread_local_tail = p;
  }
  pthread_rwlock_unlock(&lock);
#ifdef DYLINK_DEBUG
  _emscripten_errf("%p: _emscripten_thread_sync_code done", pthread_self());
#endif
  return true;
}

// This structure exists so that we can stash the status of work we do
// during emscripten_proxy_async_with_callback.
// If the function that emscripten_proxy_async_with_callback ran included
// a return value we could avoid this struct and the malloc/free of it.
// See https://github.com/emscripten-core/emscripten/issues/18378
struct thread_sync_data {
  int promise_id;
  bool result;
};

static void do_thread_sync(void* arg) {
  struct thread_sync_data* data = (struct thread_sync_data*)arg;
#ifdef DYLINK_DEBUG
  _emscripten_errf("%p: do_thread_sync: %d", pthread_self(), data->promise_id);
#endif
  data->result = _emscripten_thread_sync_code();
}

// Called once _emscripten_proxy_sync_code completes
static void done_thread_sync(void* arg) {
  struct thread_sync_data* data = (struct thread_sync_data*)arg;
#ifdef DYLINK_DEBUG
  _emscripten_errf("%p: done_thread_sync: promise_id=%d success=%d",
                   pthread_self(),
                   data->promise_id,
                   data->result);
#endif
  if (data->result) {
    emscripten_promise_resolve(data->promise_id, NULL);
  } else {
#if ABORT_ON_SYNC_FAILURE
    abort();
#else
    emscripten_promise_reject(data->promise_id);
#endif
  }
  free(data);
}

// Proxying queue specically for handling code loading (dlopen) events.
// Initialized on first call to `_emscripten_proxy_sync_code` below.
static em_proxying_queue * _Atomic dlopen_proxying_queue = NULL;

void _emscripten_process_dlopen_queue() {
  if (dlopen_proxying_queue) {
    assert(!emscripten_is_main_runtime_thread());
    emscripten_proxy_execute_queue(dlopen_proxying_queue);
  }
}

// Asynchronously runs _emscripten_thread_sync_code on the target then and
// resolves (or rejects) the given promise once it is complete.
// This function should only ever be called my the main runtime thread which
// manages the worker pool.
int _emscripten_proxy_sync_code_async(void* user_data, int promise_id) {
  pthread_t target_thread = (pthread_t)user_data;
  assert(emscripten_is_main_runtime_thread());
  if (!dlopen_proxying_queue) {
    dlopen_proxying_queue = em_proxying_queue_create();
  }
  struct thread_sync_data* data = malloc(sizeof(struct thread_sync_data));
  data->promise_id = promise_id;
  data->result = false;
  return emscripten_proxy_async_with_callback(dlopen_proxying_queue,
                                              target_thread,
                                              do_thread_sync,
                                              data,
                                              done_thread_sync,
                                              data);
}

int _emscripten_proxy_sync_code(pthread_t target_thread) {
  assert(emscripten_is_main_runtime_thread());
  if (!dlopen_proxying_queue) {
    dlopen_proxying_queue = em_proxying_queue_create();
  }
  struct thread_sync_data data = {0, 0};
  if (!emscripten_proxy_sync(
        dlopen_proxying_queue, target_thread, do_thread_sync, &data)) {
    return 0;
  }
  return data.result;
}

static void done_sync_all(em_proxying_ctx* ctx) {
#ifdef DYLINK_DEBUG
  _emscripten_errf("%p: done_sync_all", pthread_self());
#endif
  emscripten_proxy_finish(ctx);
}

static void main_thread_sync_all(em_proxying_ctx* ctx, void* arg) {
  pthread_t calling_thread = (pthread_t)arg;
#ifdef DYLINK_DEBUG
  _emscripten_errf("%p: main_thread_sync_all calling=%p", pthread_self(), calling_thread);
#endif
  _emscripten_sync_all_threads_async(calling_thread, done_sync_all, ctx);
}

static void sync_code_all_threads() {
  // Call `main_thread_sync_all` on the main thread and block until its
  // complete. This gets called after a shared library is loaded by a worker.
  pthread_t main_thread = emscripten_main_browser_thread_id();
#ifdef DYLINK_DEBUG
  _emscripten_errf("%p: sync_code_all_threads main=%p", pthread_self(), main_thread);
#endif
  if (pthread_self() == main_thread) {
    // sync_code_all_threads is called the the main thread call synchronous
    // version of emscripten_sync_all_threads
    _emscripten_sync_all_threads();
  } else {
    // Otherwise we block here while the asynchronous version runs in the main
    // thread.
    em_proxying_queue* q = emscripten_proxy_get_system_queue();
    int success = emscripten_proxy_sync_with_ctx(
      q, main_thread, main_thread_sync_all, pthread_self());
    assert(success);
  }
}
#endif // _REENTRANT

static void dlopen_onsuccess(struct dso* dso, void* user_data) {
  struct async_data* data = (struct async_data*)user_data;
#ifdef DYLINK_DEBUG
  _emscripten_errf("%p: dlopen_js_onsuccess: dso=%p mem_addr=%p mem_size=%zu",
                   pthread_self(),
                   dso,
                   dso->mem_addr,
                   dso->mem_size);
#endif
  load_library_done(dso);
  pthread_rwlock_unlock(&lock);
#ifdef _REENTRANT
  // Block until all other threads have loaded this module.
  sync_code_all_threads();
#endif
  pthread_mutex_unlock(&dlopen_mutex);
  data->onsuccess(data->user_data, dso);
  free(data);
}

static void dlopen_onerror(struct dso* dso, void* user_data) {
  struct async_data* data = (struct async_data*)user_data;
#ifdef DYLINK_DEBUG
  _emscripten_errf("%p: dlopen_js_onerror: dso=%p", pthread_self(), dso);
#endif
  pthread_rwlock_unlock(&lock);
  pthread_mutex_unlock(&dlopen_mutex);
  data->onerror(data->user_data);
  free(dso);
  free(data);
}

void* dlopen(const char* file, int flags) {
  ensure_init();
  if (!file) {
    return head;
  }

  // First grab the outer lock which protects both the local dlopen process on
  // the `sync_code_all_threads`.  This means that other threads can't start to
  // load more dlls before the `sync_code_all_threads` proccess completes.
  pthread_mutex_lock(&dlopen_mutex);

#ifdef DYLINK_DEBUG
  _emscripten_errf("%p: dlopen: %s [%d]", pthread_self(), file, flags);
#endif

  struct dso* p;
  pthread_rwlock_wrlock(&lock);

  /* Search for the name to see if it's already loaded */
  for (p = head; p; p = p->next) {
    if (!strcmp(p->name, file)) {
#ifdef DYLINK_DEBUG
      _emscripten_errf("%p: dlopen: already opened: %p", pthread_self(), p);
#endif
      pthread_rwlock_unlock(&lock);
      pthread_mutex_unlock(&dlopen_mutex);
      return p;
    }
  }

  int cs;
  pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &cs);
  p = load_library_start(file, flags);
  if (!p) {
    goto error;
  }
  void* success = _dlopen_js(p);
  if (!success) {
#ifdef DYLINK_DEBUG
    _emscripten_errf("%p: dlopen_js: failed: %p", pthread_self(), p);
#endif
    free(p);
    goto error;
  }
  load_library_done(p);
  pthread_rwlock_unlock(&lock);
#ifdef _REENTRANT
  // Block until all other threads have loaded this module.
  sync_code_all_threads();
#endif
#ifdef DYLINK_DEBUG
  _emscripten_errf("%p: dlopen(%s): success: %p", pthread_self(), file, p);
#endif
  pthread_mutex_unlock(&dlopen_mutex);
  pthread_setcancelstate(cs, 0);
  return p;

error:
  pthread_rwlock_unlock(&lock);
  pthread_mutex_unlock(&dlopen_mutex);
  pthread_setcancelstate(cs, 0);
  return NULL;
}

void emscripten_dlopen(const char* filename, int flags, void* user_data,
                       em_dlopen_callback onsuccess, em_arg_callback_func onerror) {
  ensure_init();
  if (!filename) {
    onsuccess(user_data, head);
    return;
  }
  pthread_mutex_lock(&dlopen_mutex);
  pthread_rwlock_wrlock(&lock);
  struct dso* p = load_library_start(filename, flags);
  if (!p) {
    pthread_rwlock_unlock(&lock);
    pthread_mutex_unlock(&dlopen_mutex);
    onerror(user_data);
    return;
  }

  // For async mode
  struct async_data* d = malloc(sizeof(struct async_data));
  d->user_data = user_data;
  d->onsuccess = onsuccess;
  d->onerror = onerror;

#ifdef DYLINK_DEBUG
  _emscripten_errf("%p: calling emscripten_dlopen_js %p", pthread_self(), p);
#endif
  // Unlock happens in dlopen_onsuccess/dlopen_onerror
  _emscripten_dlopen_js(p, dlopen_onsuccess, dlopen_onerror, d);
}

void* __dlsym(void* restrict p, const char* restrict s, void* restrict ra) {
#ifdef DYLINK_DEBUG
  _emscripten_errf("%p: __dlsym dso:%p sym:%s", pthread_self(), p, s);
#endif
  if (p != RTLD_DEFAULT && p != RTLD_NEXT && __dl_invalid_handle(p)) {
    return 0;
  }
  void* res;
  pthread_rwlock_rdlock(&lock);
  res = _dlsym_js(p, s);
  pthread_rwlock_unlock(&lock);
  return res;
}

int dladdr(const void* addr, Dl_info* info) {
  // report all function pointers as coming from this program itself XXX not
  // really correct in any way
  info->dli_fname = "unknown";
  info->dli_fbase = NULL;
  info->dli_sname = NULL;
  info->dli_saddr = NULL;
  return 1;
}
