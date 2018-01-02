#ifndef CERT_TRANS_UTIL_THREAD_POOL_H_
#define CERT_TRANS_UTIL_THREAD_POOL_H_

#include <chrono>
#include <functional>
#include <map>
#include <memory>

#include "base/macros.h"
#include "util/executor.h"

namespace cert_trans {


// Provides a fixed size thread pool to run closures on. The pool is
// sized according to the number of cores in the system.
class ThreadPool : public util::Executor {
 public:
  // Creates the threads.
  ThreadPool();

  // Creates the threads.
  ThreadPool(size_t num_threads);

  // The destructor will wait for any outstanding closures to finish.
  ~ThreadPool();

  // Arranges for "closure" to be called in the thread pool. The
  // function must not be empty.
  void Add(const std::function<void()>& closure) override;

  void Delay(const std::chrono::duration<double>& delay,
             util::Task* task) override;

 private:
  class Impl;
  const std::unique_ptr<Impl> impl_;

  DISALLOW_COPY_AND_ASSIGN(ThreadPool);
};


}  // namespace cert_trans

#endif  // CERT_TRANS_UTIL_THREAD_POOL_H_
