#include "util/openssl_util.h"

#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/pem.h>
#include <string>

using std::string;

namespace util {

string DumpOpenSSLErrorStack() {
  if (ERR_peek_error() == 0)
    return "No OpenSSL errors left on stack";

  string stack_dump("OpenSSL errors left on stack:");
  unsigned long error;
  char error_string[256];
  while ((error = ERR_get_error()) != 0) {
    stack_dump.append("\n\t");
    ERR_error_string_n(error, error_string, 256);
    stack_dump.append(error_string);
  }
  return stack_dump;
}

void ClearOpenSSLErrors() {
  ERR_clear_error();
}

string ReadBIO(BIO* bio) {
  int size = BIO_pending(bio);
  char* buffer = new char[size];
  int bytes_read = BIO_read(bio, buffer, size);
  if (bytes_read != size) {
    LOG(ERROR) << "Read " << bytes_read << " bytes; expected " << size;
    delete[] buffer;
    return string();
  }

  string ret(buffer, bytes_read);
  delete[] buffer;
  return ret;
}

}  // namespace util
