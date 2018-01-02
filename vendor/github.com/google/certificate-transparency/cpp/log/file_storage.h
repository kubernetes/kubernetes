#ifndef CERT_TRANS_LOG_FILE_STORAGE_H_
#define CERT_TRANS_LOG_FILE_STORAGE_H_

#include <stdint.h>
#include <memory>
#include <set>
#include <string>

#include "base/macros.h"
#include "util/status.h"

namespace cert_trans {

class FilesystemOps;

// A simple filesystem-based database for (key, data) entries,
// structured as follows:
//
// <root>/storage - Storage for the data, filenames are derived from
//                  the hex key like so: "89abcd" becomes "8/9/a/bcd"
//                  (for storage depth 3). This is because filesystems
//                  tend to perform badly with very large
//                  directories. For this to work, we assume keys are
//                  hashes, i.e., random, and of reasonable
//                  length. However, numerical monotonically
//                  increasing keys can be made to work too: for
//                  example, 4-byte keys could be set up with max 256
//                  entries/directory by setting storage_depth=6:
//
//                  "00000000" -> "0/0/0/0/0/0/00"
//                  "00000001" -> "0/0/0/0/0/0/01"
//                  ...
//                  "00000100" -> "0/0/0/0/0/1/00"
//                  ...
//
//                  Each key corresponds to a file with the
//                  data. Writes to these files are atomic
//                  (i.e. create a new file and move into place).
//
// <root>/tmp     - Temporary storage for atomicity. Must be on the
//                  same filesystem as <root>/storage.
//
// FileStorage aborts upon any FilesystemOps error. This class is
// threadsafe.
class FileStorage {
 public:
  // Default constructor, uses BasicFilesystemOps.
  FileStorage(const std::string& file_base, int storage_depth);
  // Takes ownership of the FilesystemOps.
  FileStorage(const std::string& file_base, int storage_depth,
              cert_trans::FilesystemOps* file_op);
  ~FileStorage();

  // Scan the entire database and return the list of keys.
  std::set<std::string> Scan() const;

  // Write (key, data) unless an entry matching |key| already exists.
  util::Status CreateEntry(const std::string& key, const std::string& data);

  // Update an existing entry; fail if it doesn't already exist.
  util::Status UpdateEntry(const std::string& key, const std::string& data);

  // Lookup entry based on key.
  util::Status LookupEntry(const std::string& key, std::string* result) const;

 private:
  std::string StoragePathBasename(const std::string& hex) const;
  std::string StoragePathComponent(const std::string& hex, int n) const;
  std::string StoragePath(const std::string& key) const;
  std::string StorageKey(const std::string& storage_path) const;
  // Write or overwrite.
  void WriteStorageEntry(const std::string& key, const std::string& data);
  void ScanFiles(const std::string& dir_path,
                 std::set<std::string>* keys) const;
  void ScanDir(const std::string& dir_path, int depth,
               std::set<std::string>* keys) const;

  // The following methods abort upon any error.
  bool FileExists(const std::string& file_path) const;
  void AtomicWriteBinaryFile(const std::string& file_path,
                             const std::string& data);
  // Create directory, unless it already exists.
  void CreateMissingDirectory(const std::string& dir_path);

  const std::string storage_dir_;
  const std::string tmp_dir_;
  const std::string tmp_file_template_;
  const int storage_depth_;
  const std::unique_ptr<cert_trans::FilesystemOps> file_op_;

  DISALLOW_COPY_AND_ASSIGN(FileStorage);
};


}  // namespace cert_trans

#endif  // CERT_TRANS_LOG_FILE_STORAGE_H_
