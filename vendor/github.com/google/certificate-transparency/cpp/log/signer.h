// A base class for signing unstructured data.  This class is mockable.

#ifndef SRC_LOG_SIGNER_H_
#define SRC_LOG_SIGNER_H_

#include <openssl/evp.h>
#include <openssl/x509.h>  // for i2d_PUBKEY
#include <stdint.h>

#include "base/macros.h"
#include "proto/ct.pb.h"
#include "util/openssl_scoped_types.h"

namespace cert_trans {

class Signer {
 public:
  explicit Signer(EVP_PKEY* pkey);
  virtual ~Signer() = default;

  virtual std::string KeyID() const;

  virtual void Sign(const std::string& data,
                    ct::DigitallySigned* signature) const;

 protected:
  // A constructor for mocking.
  Signer();

 private:
  std::string RawSign(const std::string& data) const;

  ScopedEVP_PKEY pkey_;
  ct::DigitallySigned::HashAlgorithm hash_algo_;
  ct::DigitallySigned::SignatureAlgorithm sig_algo_;
  std::string key_id_;

  DISALLOW_COPY_AND_ASSIGN(Signer);
};

}  // namespace cert_trans

#endif  // SRC_LOG_SIGNER_H_
