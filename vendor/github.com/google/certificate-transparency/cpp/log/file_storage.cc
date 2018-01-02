/* -*- indent-tabs-mode: nil -*- */
#include "log/file_storage.h"

#include <dirent.h>
#include <errno.h>
#include <glog/logging.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>
#include <set>
#include <string>

#include "log/filesystem_ops.h"
#include "util/util.h"

using cert_trans::BasicFilesystemOps;
using cert_trans::FilesystemOps;
using std::string;

namespace cert_trans {


FileStorage::FileStorage(const string& file_base, int storage_depth)
    : storage_dir_(file_base + "/storage"),
      tmp_dir_(file_base + "/tmp"),
      tmp_file_template_(tmp_dir_ + "/tmpXXXXXX"),
      storage_depth_(storage_depth),
      file_op_(new BasicFilesystemOps()) {
  CHECK_GE(storage_depth_, 0);
  CreateMissingDirectory(storage_dir_);
  CreateMissingDirectory(tmp_dir_);
}


FileStorage::FileStorage(const string& file_base, int storage_depth,
                         FilesystemOps* file_op)
    : storage_dir_(file_base + "/storage"),
      tmp_dir_(file_base + "/tmp"),
      tmp_file_template_(tmp_dir_ + "/tmpXXXXXX"),
      storage_depth_(storage_depth),
      file_op_(CHECK_NOTNULL(file_op)) {
  CHECK_GE(storage_depth_, 0);
  CreateMissingDirectory(storage_dir_);
  CreateMissingDirectory(tmp_dir_);
}


FileStorage::~FileStorage() {
  // Needs to be where FilesystemOps is visible.
}


std::set<string> FileStorage::Scan() const {
  std::set<string> storage_keys;
  ScanDir(storage_dir_, storage_depth_, &storage_keys);
  return storage_keys;
}


util::Status FileStorage::CreateEntry(const string& key, const string& data) {
  if (LookupEntry(key, NULL).ok()) {
    return util::Status(util::error::ALREADY_EXISTS,
                        "entry already exists: " + key);
  }
  WriteStorageEntry(key, data);
  return util::Status::OK;
}


util::Status FileStorage::UpdateEntry(const string& key, const string& data) {
  if (!LookupEntry(key, NULL).ok()) {
    return util::Status(util::error::NOT_FOUND,
                        "tried to update non-existent entry: " + key);
  }
  WriteStorageEntry(key, data);
  return util::Status::OK;
}


util::Status FileStorage::LookupEntry(const string& key,
                                      string* result) const {
  string data_file = StoragePath(key);
  if (!FileExists(data_file)) {
    return util::Status(util::error::NOT_FOUND, "entry not found: " + key);
  }
  if (result) {
    CHECK(util::ReadBinaryFile(data_file, result));
  }
  return util::Status::OK;
}


string FileStorage::StoragePathBasename(const string& hex) const {
  if (hex.length() <= static_cast<uint>(storage_depth_))
    return "-";
  return hex.substr(storage_depth_);
}


string FileStorage::StoragePathComponent(const string& hex, int n) const {
  CHECK_GE(n, 0);
  CHECK_LT(n, storage_depth_);
  if (static_cast<uint>(n) >= hex.length())
    return "-";
  return string(1, hex[n]);
}


string FileStorage::StoragePath(const string& key) const {
  string hex = util::HexString(key);
  string dirname = storage_dir_ + "/";
  for (int n = 0; n < storage_depth_; ++n)
    dirname += StoragePathComponent(hex, n) + "/";
  return dirname + StoragePathBasename(hex);
}


string FileStorage::StorageKey(const string& storage_path) const {
  CHECK_EQ(storage_path.substr(0, storage_dir_.size()), storage_dir_);
  string key_path = storage_path.substr(storage_dir_.size() + 1);
  string hex_key;
  for (int n = 0; n < storage_depth_; ++n) {
    char hex_char = key_path[2 * n];
    if (hex_char == '-')
      return util::BinaryString(hex_key);
    hex_key.push_back(hex_char);
  }
  string basename = key_path.substr(2 * storage_depth_);
  if (basename != "-")
    hex_key.append(basename);
  return util::BinaryString(hex_key);
}


void FileStorage::WriteStorageEntry(const string& key, const string& data) {
  string hex = util::HexString(key);

  // Make the intermediate directories, if needed.
  // TODO(ekasper): we can skip this if we know we're updating.
  string dir = storage_dir_;
  for (int n = 0; n < storage_depth_; ++n) {
    dir += "/" + StoragePathComponent(hex, n);
    CreateMissingDirectory(dir);
  }

  // == StoragePath(key)
  string filename = dir + "/" + StoragePathBasename(hex);
  AtomicWriteBinaryFile(filename, data);
}


void FileStorage::ScanFiles(const string& dir_path,
                            std::set<string>* keys) const {
  DIR* dir = CHECK_NOTNULL(opendir(dir_path.c_str()));
  struct dirent* entry;
  while ((entry = readdir(dir)) != NULL) {
    if (entry->d_name[0] == '.')
      continue;
    keys->insert(StorageKey(dir_path + "/" + entry->d_name));
  }
  closedir(dir);
}


void FileStorage::ScanDir(const string& dir_path, int depth,
                          std::set<string>* keys) const {
  CHECK_GE(depth, 0);
  if (depth > 0) {
    // Parse subdirectories. (TODO: make opendir part of filesystemop).
    DIR* dir = CHECK_NOTNULL(opendir(dir_path.c_str()));
    struct dirent* entry;
    std::set<string> result;
    while ((entry = readdir(dir)) != NULL) {
      if (entry->d_name[0] == '.')
        continue;
      ScanDir(dir_path + "/" + entry->d_name, depth - 1, keys);
    }
    closedir(dir);
  } else {
    // depth == 0; parse files.
    ScanFiles(dir_path, keys);
  }
}


bool FileStorage::FileExists(const string& file_path) const {
  if (file_op_->access(file_path, F_OK) == 0)
    return true;

  CHECK_EQ(errno, ENOENT);

  return false;
}


void FileStorage::AtomicWriteBinaryFile(const string& file_path,
                                        const string& data) {
  const string tmp_file(
      util::WriteTemporaryBinaryFile(tmp_file_template_, data));

  CHECK(!tmp_file.empty());
  CHECK_EQ(file_op_->rename(tmp_file, file_path), 0);
}


void FileStorage::CreateMissingDirectory(const string& dir_path) {
  if (file_op_->mkdir(dir_path, 0700) != 0) {
    CHECK_EQ(errno, EEXIST);
  }
}


}  // namespace cert_trans
