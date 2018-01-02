#ifndef CERT_TRANS_UTIL_EXECUTOR_H_
#define CERT_TRANS_UTIL_EXECUTOR_H_

#include <chrono>
#include <functional>

#include "base/macros.h"

namespace util {
class Task;


class Executor {
 public:
  virtual ~Executor() = default;

  virtual void Add(const std::function<void()>& closure) = 0;
  virtual void Delay(const std::chrono::duration<double>& delay,
                     Task* task) = 0;

 protected:
  Executor() = default;

 private:
  DISALLOW_COPY_AND_ASSIGN(Executor);
};


}  // namespace util

#endif  // CERT_TRANS_UTIL_EXECUTOR_H_
