#include "util/read_key.h"

#include <openssl/pem.h>
#include <memory>

using std::unique_ptr;

namespace cert_trans {

namespace {


void FileCloser(FILE* fp) {
  if (fp) {
    fclose(fp);
  }
}


}  // namespace


util::StatusOr<EVP_PKEY*> ReadPrivateKey(const std::string& file) {
  unique_ptr<FILE, void (*)(FILE*)> fp(fopen(file.c_str(), "r"), FileCloser);

  if (!fp) {
    return util::Status(util::error::NOT_FOUND, "key file not found: " + file);
  }

  // No password.
  EVP_PKEY* retval(nullptr);
  PEM_read_PrivateKey(fp.get(), &retval, nullptr, nullptr);
  if (!retval)
    return util::Status(util::error::FAILED_PRECONDITION,
                        "invalid key: " + file);

  return retval;
}


util::StatusOr<EVP_PKEY*> ReadPublicKey(const std::string& file) {
  unique_ptr<FILE, void (*)(FILE*)> fp(fopen(file.c_str(), "r"), FileCloser);

  if (!fp) {
    return util::Status(util::error::NOT_FOUND, "key file not found: " + file);
  }

  // No password.
  EVP_PKEY* retval(nullptr);
  PEM_read_PUBKEY(fp.get(), &retval, nullptr, nullptr);
  if (!retval)
    return util::Status(util::error::FAILED_PRECONDITION,
                        "invalid key: " + file);

  return retval;
}


}  // namespace cert_trans
