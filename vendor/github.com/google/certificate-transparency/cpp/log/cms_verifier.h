/* -*- mode: c++; indent-tabs-mode: nil -*- */
#ifndef CMS_VERIFIER_H
#define CMS_VERIFIER_H

#include <openssl/asn1.h>
#include <openssl/bio.h>
#include <openssl/cms.h>
#include <memory>

#include "base/macros.h"
#include "log/cert.h"
#include "util/openssl_util.h"  // for LOG_OPENSSL_ERRORS
#include "util/status.h"

namespace cert_trans {

class CmsVerifier {
 public:
  CmsVerifier() = default;

  virtual ~CmsVerifier() = default;

  // NOTE: CMS related API is provisional and may evolve over the near
  // future. Public API does not refer to OpenSSL CMS data objects to
  // allow for future use with alternate S/MIME implementations providing
  // CMS functionality.

  // Checks that a CMS_ContentInfo has a signer that matches a specified
  // certificate. Does not verify the signature or check the payload.
  virtual util::StatusOr<bool> IsCmsSignedByCert(BIO* cms_bio_in,
                                                 const Cert& cert) const;
  // Checks that a CMS_ContentInfo has a signer that matches a specified
  // certificate. Does not verify the signature or check the payload.
  virtual util::StatusOr<bool> IsCmsSignedByCert(const std::string& cms_object,
                                                 const Cert* cert) const;

  // Unpacks a CMS signed data object that is assumed to contain a certificate
  // Does not do any checks on signatures or cert validity at this point,
  // the caller must do these separately. Returns a new Cert object built from
  // the unpacked data, which will only be valid if we successfully unpacked
  // the CMS blob.
  virtual Cert* UnpackCmsSignedCertificate(const std::string& cms_object);

  // Unpacks a CMS signed data object that is assumed to contain a certificate
  // If the CMS signature verifies as being signed by the supplied Cert
  // then we return a corresponding new Cert object built from the unpacked
  // data. If it cannot be loaded as a certificate or fails CMS signing check
  // then an unloaded empty Cert object is returned.
  // The caller owns the returned certificate and must free the input bio.
  // NOTE: Certificate validity checks must be done separately. This
  // only checks that the CMS signature is validly made by the supplied
  // certificate.
  virtual Cert* UnpackCmsSignedCertificate(BIO* cms_bio_in,
                                           const Cert& verify_cert);

 private:
  // Verifies that data from a DER BIO is signed by a given certificate.
  // and writes the unwrapped content to another BIO. NULL can be passed for
  // cms_bio_out if the caller just wishes to verify the signature. Does
  // not free either BIO. Does not do any checks on the content of the
  // CMS message or validate that the CMS signature is trusted to root.
  util::Status UnpackCmsDerBio(BIO* cms_bio_in, const Cert& certChain,
                               BIO* cms_bio_out);
  // Writes the unwrapped content from a CMS object to another BIO. Does
  // not free either BIO. Does not do any checks on the content of the
  // CMS message or validate that the CMS signature is trusted to root.
  // The unpacked data may not be a valid X.509 cert. The caller must
  // apply any additional checks necessary.
  util::Status UnpackCmsDerBio(BIO* cms_bio_in, BIO* cms_bio_out);

  DISALLOW_COPY_AND_ASSIGN(CmsVerifier);
};

}  // namespace cert_trans
#endif
