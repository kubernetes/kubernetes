#ifndef CERT_TRANS_LOG_FILESYSTEM_OPS_H_
#define CERT_TRANS_LOG_FILESYSTEM_OPS_H_

#include <sys/types.h>
#include <string>

#include "base/macros.h"

namespace cert_trans {


// Make filesystem operations virtual so that we can override
// to simulate filesystem errors.
class FilesystemOps {
 public:
  virtual ~FilesystemOps() = default;

  virtual int mkdir(const std::string& path, mode_t mode) = 0;
  virtual int remove(const std::string& path) = 0;
  virtual int rename(const std::string& old_name,
                     const std::string& new_name) = 0;
  virtual int access(const std::string& path, int amode) = 0;

 protected:
  FilesystemOps() = default;

 private:
  DISALLOW_COPY_AND_ASSIGN(FilesystemOps);
};


class BasicFilesystemOps : public FilesystemOps {
 public:
  BasicFilesystemOps() = default;

  int mkdir(const std::string& path, mode_t mode) override;
  int remove(const std::string& path) override;
  int rename(const std::string& old_name,
             const std::string& new_name) override;
  int access(const std::string& path, int amode) override;
};


// Fail at an operation with a given op count.
class FailingFilesystemOps : public BasicFilesystemOps {
 public:
  explicit FailingFilesystemOps(int fail_point);

  int OpCount() const {
    return op_count_;
  }

  int mkdir(const std::string& path, mode_t mode) override;
  int remove(const std::string& path) override;
  int rename(const std::string& old_name,
             const std::string& new_name) override;
  int access(const std::string& path, int amode) override;

 private:
  int op_count_;
  int fail_point_;
};


}  // namespace cert_trans

#endif  // CERT_TRANS_LOG_FILESYSTEM_OPS_H_
