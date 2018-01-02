#include "log/filesystem_ops.h"

#include <errno.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

namespace cert_trans {


int BasicFilesystemOps::mkdir(const std::string& path, mode_t mode) {
  return ::mkdir(path.c_str(), mode);
}


int BasicFilesystemOps::remove(const std::string& path) {
  return ::remove(path.c_str());
}


int BasicFilesystemOps::rename(const std::string& old_name,
                               const std::string& new_name) {
  return ::rename(old_name.c_str(), new_name.c_str());
}


int BasicFilesystemOps::access(const std::string& path, int amode) {
  return ::access(path.c_str(), amode);
}


FailingFilesystemOps::FailingFilesystemOps(int fail_point)
    : op_count_(0), fail_point_(fail_point) {
}


int FailingFilesystemOps::mkdir(const std::string& path, mode_t mode) {
  if (fail_point_ == op_count_++) {
    errno = EIO;
    return -1;
  }
  return BasicFilesystemOps::mkdir(path, mode);
}


int FailingFilesystemOps::remove(const std::string& path) {
  if (fail_point_ == op_count_++) {
    errno = EIO;
    return -1;
  }
  return BasicFilesystemOps::remove(path);
}


int FailingFilesystemOps::rename(const std::string& old_name,
                                 const std::string& new_name) {
  if (fail_point_ == op_count_++) {
    errno = EIO;
    return -1;
  }
  return BasicFilesystemOps::rename(old_name, new_name);
}


int FailingFilesystemOps::access(const std::string& path, int amode) {
  if (fail_point_ == op_count_++) {
    errno = EACCES;
    return -1;
  }
  return BasicFilesystemOps::access(path, amode);
}


}  // namespace cert_trans
