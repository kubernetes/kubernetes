/* -*- indent-tabs-mode: nil -*- */
#include "log/cert.h"
#include "log/ct_extensions.h"
#include "merkletree/serial_hasher.h"
#include "util/openssl_util.h"  // For LOG_OPENSSL_ERRORS
#include "util/util.h"

#include <glog/logging.h>
#include <openssl/asn1.h>
#include <openssl/bio.h>
#include <openssl/crypto.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/objects.h>
#include <openssl/pem.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <time.h>
#include <memory>
#include <string>
#include <vector>

using std::string;
using std::to_string;
using std::unique_ptr;
using std::vector;
using util::ClearOpenSSLErrors;
using util::StatusOr;
using util::error::Code;

#if OPENSSL_VERSION_NUMBER < 0x10002000L || defined(OPENSSL_IS_BORINGSSL)
// Backport from 1.0.2-beta3.
static int i2d_re_X509_tbs(X509* x, unsigned char** pp) {
  x->cert_info->enc.modified = 1;
  return i2d_X509_CINF(x->cert_info, pp);
}
#endif

#if OPENSSL_VERSION_NUMBER < 0x10002000L
static int X509_get_signature_nid(const X509* x) {
  return OBJ_obj2nid(x->sig_alg->algorithm);
}
#endif


namespace cert_trans {


// Convert string from ASN1 and check it doesn't contain nul characters
string ASN1ToStringAndCheckForNulls(ASN1_STRING* asn1_string,
                                    const string& tag, util::Status* status) {
  const string cpp_string(reinterpret_cast<char*>(
                              ASN1_STRING_data(asn1_string)),
                          ASN1_STRING_length(asn1_string));

  // Unfortunately ASN1_STRING_length returns a signed value
  if (ASN1_STRING_length(asn1_string) < 0) {
    *status = util::Status(Code::INVALID_ARGUMENT, "ASN1 string is corrupt?");
  }

  // Make sure there isn't an embedded NUL character in the DNS ID
  // We now know it's not a negative length so this can't overflow.
  if (static_cast<size_t>(ASN1_STRING_length(asn1_string)) !=
      cpp_string.length()) {
    LOG(ERROR) << "Embedded null in asn1 string: " << tag;
    *status =
        util::Status(Code::INVALID_ARGUMENT, "Embedded null in asn1 string");
  } else {
    *status = util::Status::OK;
  }

  return cpp_string;
}


Cert::Cert(X509* x509) : x509_(x509) {
}


Cert::Cert(const string& pem_string) {
  // A read-only bio.
  ScopedBIO bio_in(BIO_new_mem_buf(const_cast<char*>(pem_string.data()),
                                   pem_string.length()));
  if (!bio_in) {
    LOG_OPENSSL_ERRORS(ERROR);
    return;
  }

  x509_.reset(PEM_read_bio_X509(bio_in.get(), nullptr, nullptr, nullptr));

  if (!x509_) {
    // At this point most likely the input was just corrupt. There are a few
    // real errors that may have happened (a malloc failure is one) and it is
    // virtually impossible to fish them out.
    LOG(WARNING) << "Input is not a valid PEM-encoded certificate";
    LOG_OPENSSL_ERRORS(WARNING);
  }
}


Cert* Cert::Clone() const {
  X509* x509(nullptr);
  if (x509_) {
    x509 = X509_dup(x509_.get());
    if (!x509)
      LOG_OPENSSL_ERRORS(ERROR);
  }
  return new Cert(x509);
}


util::Status Cert::LoadFromDerString(const string& der_string) {
  const unsigned char* start =
      reinterpret_cast<const unsigned char*>(der_string.data());
  x509_.reset(d2i_X509(nullptr, &start, der_string.size()));
  if (!x509_) {
    LOG(WARNING) << "Input is not a valid DER-encoded certificate";
    LOG_OPENSSL_ERRORS(WARNING);
    return util::Status(Code::INVALID_ARGUMENT, "Not a valid encoded cert");
  }
  return util::Status::OK;
}


util::Status Cert::LoadFromDerBio(BIO* bio_in) {
  x509_.reset(d2i_X509_bio(bio_in, nullptr));
  CHECK_NOTNULL(bio_in);

  if (!x509_) {
    // At this point most likely the input was just corrupt. There are few
    // real errors that may have happened (a malloc failure is one) and it is
    // virtually impossible to fish them out.
    LOG(WARNING) << "Input is not a valid encoded certificate";
    LOG_OPENSSL_ERRORS(WARNING);
    return util::Status(Code::INVALID_ARGUMENT, "Not a valid encoded cert");
  }
  return util::Status::OK;
}


string Cert::PrintIssuerName() const {
  if (!IsLoaded()) {
    LOG(ERROR) << "Cert not loaded";
    return string();
  }

  return PrintName(X509_get_issuer_name(x509_.get()));
}


string Cert::PrintSubjectName() const {
  if (!IsLoaded()) {
    LOG(ERROR) << "Cert not loaded";
    return string();
  }

  return PrintName(X509_get_subject_name(x509_.get()));
}


// static
string Cert::PrintName(X509_NAME* name) {
  if (!name)
    return string();
  ScopedBIO bio(BIO_new(BIO_s_mem()));
  if (!bio) {
    LOG_OPENSSL_ERRORS(ERROR);
    return string();
  }

  if (X509_NAME_print_ex(bio.get(), name, 0, 0) != 1) {
    LOG_OPENSSL_ERRORS(ERROR);
    return string();
  }

  string ret = util::ReadBIO(bio.get());
  return ret;
}


string Cert::PrintNotBefore() const {
  if (!IsLoaded()) {
    LOG(ERROR) << "Cert not loaded";
    return string();
  }

  return PrintTime(X509_get_notBefore(x509_.get()));
}


string Cert::PrintNotAfter() const {
  if (!IsLoaded()) {
    LOG(ERROR) << "Cert not loaded";
    return string();
  }

  return PrintTime(X509_get_notAfter(x509_.get()));
}


string Cert::PrintSignatureAlgorithm() const {
  const char* sigalg = OBJ_nid2ln(X509_get_signature_nid(x509_.get()));
  if (!sigalg)
    return "NULL";
  return string(sigalg);
}


// static
string Cert::PrintTime(ASN1_TIME* when) {
  if (!when)
    return string();

  ScopedBIO bio(BIO_new(BIO_s_mem()));
  if (!bio) {
    LOG_OPENSSL_ERRORS(ERROR);
    return string();
  }

  if (ASN1_TIME_print(bio.get(), when) != 1) {
    LOG_OPENSSL_ERRORS(ERROR);
    return string();
  }

  string ret = util::ReadBIO(bio.get());
  return ret;
}


bool Cert::IsIdenticalTo(const Cert& other) const {
  return X509_cmp(x509_.get(), other.x509_.get()) == 0;
}


util::StatusOr<bool> Cert::HasExtension(int extension_nid) const {
  if (!IsLoaded()) {
    LOG(ERROR) << "Cert not loaded";
    return util::Status(Code::FAILED_PRECONDITION, "Cert not loaded");
  }

  const StatusOr<int> index(ExtensionIndex(extension_nid));
  if (index.ok()) {
    return true;
  }

  if (index.status().CanonicalCode() == util::error::NOT_FOUND) {
    return false;
  }

  return util::Status(Code::INTERNAL, "Failed to get extension");
}


StatusOr<bool> Cert::HasCriticalExtension(int extension_nid) const {
  if (!IsLoaded()) {
    LOG(ERROR) << "Cert not loaded";
    return util::Status(Code::FAILED_PRECONDITION, "Cert not loaded");
  }

  const StatusOr<X509_EXTENSION*> ext(GetExtension(extension_nid));
  if (!ext.ok()) {
    // The extension may be absent, which is not an error
    if (ext.status().CanonicalCode() == util::error::NOT_FOUND) {
      return false;
    } else {
      return util::Status(Code::INTERNAL, "Failed to get extension");
    }
  }

  return X509_EXTENSION_get_critical(ext.ValueOrDie()) > 0;
}


StatusOr<bool> Cert::HasBasicConstraintCATrue() const {
  if (!IsLoaded()) {
    LOG(ERROR) << "Cert not loaded";
    return util::Status(Code::FAILED_PRECONDITION, "Cert not loaded");
  }

  const StatusOr<void*> ext_struct(ExtensionStructure(NID_basic_constraints));

  if (ext_struct.status().CanonicalCode() == Code::NOT_FOUND) {
    // No extension found
    return false;
  } else if (!ext_struct.ok()) {
    // Truly odd.
    LOG(ERROR) << "Failed to check BasicConstraints extension";
    return ext_struct.status();
  }

  // |constraints| is never null upon success.
  ScopedBASIC_CONSTRAINTS basic_constraints(
      static_cast<BASIC_CONSTRAINTS*>(ext_struct.ValueOrDie()));
  bool is_ca = basic_constraints->ca;
  return is_ca;
}


StatusOr<bool> Cert::HasExtendedKeyUsage(int key_usage_nid) const {
  if (!IsLoaded()) {
    LOG(ERROR) << "Cert not loaded";
    return util::Status(Code::FAILED_PRECONDITION, "Cert not loaded");
  }

  const ASN1_OBJECT* key_usage_obj = OBJ_nid2obj(key_usage_nid);
  if (!key_usage_obj) {
    LOG(ERROR) << "OpenSSL OBJ_nid2obj returned NULL for NID " << key_usage_nid
               << ". Is the NID not recognised?";
    LOG_OPENSSL_ERRORS(WARNING);
    return util::Status(Code::INTERNAL, "NID lookup failed");
  }

  const StatusOr<void*> ext_key_usage = ExtensionStructure(NID_ext_key_usage);

  if (ext_key_usage.status().CanonicalCode() == Code::NOT_FOUND) {
    // No extension found
    return false;
  } else if (!ext_key_usage.ok()) {
    // Truly odd.
    LOG(ERROR) << "Failed to check ExtendedKeyUsage extension";
    return ext_key_usage.status();
  }

  // |eku| is never null upon success.
  ScopedEXTENDED_KEY_USAGE eku(
      static_cast<EXTENDED_KEY_USAGE*>(ext_key_usage.ValueOrDie()));
  bool ext_key_usage_found = false;
  for (int i = 0; i < sk_ASN1_OBJECT_num(eku.get()); ++i) {
    if (OBJ_cmp(key_usage_obj, sk_ASN1_OBJECT_value(eku.get(), i)) == 0) {
      ext_key_usage_found = true;
      break;
    }
  }

  return ext_key_usage_found;
}


StatusOr<bool> Cert::IsIssuedBy(const Cert& issuer) const {
  if (!IsLoaded() || !issuer.IsLoaded()) {
    LOG(ERROR) << "Cert not loaded";
    return util::Status(Code::FAILED_PRECONDITION, "Cert not loaded");
  }
  // Seemingly no negative "real" error codes are returned from openssl api.
  return X509_check_issued(const_cast<X509*>(issuer.x509_.get()),
                           x509_.get()) == X509_V_OK;
}

StatusOr<bool> Cert::LogUnsupportedAlgorithm() const {
  LOG(WARNING) << "Unsupported algorithm: " << PrintSignatureAlgorithm();
  ClearOpenSSLErrors();
  return util::Status(Code::UNIMPLEMENTED, "Unsupported algorithm");
}

StatusOr<bool> Cert::IsSignedBy(const Cert& issuer) const {
  if (!IsLoaded() || !issuer.IsLoaded()) {
    LOG(ERROR) << "Cert not loaded";
    return util::Status(Code::FAILED_PRECONDITION, "Cert not loaded");
  }

  const ScopedEVP_PKEY issuer_key(X509_get_pubkey(issuer.x509_.get()));
  if (!issuer_key) {
    LOG(WARNING) << "NULL issuer key";
    LOG_OPENSSL_ERRORS(WARNING);
    return false;
  }

  const int ret(X509_verify(x509_.get(), issuer_key.get()));
  if (ret == 1) {
    return true;
  }
  unsigned long err = ERR_peek_last_error();
  const int reason = ERR_GET_REASON(err);
  const int lib = ERR_GET_LIB(err);
#if defined(OPENSSL_IS_BORINGSSL)
  // BoringSSL returns only 0 and 1.  This is an attempt to
  // approximate the circumstances that in OpenSSL cause a 0 return,
  // and that are too boring/spammy to log, e.g. malformed inputs.
  if (err == 0 || lib == ERR_LIB_ASN1 || lib == ERR_LIB_X509) {
    ClearOpenSSLErrors();
    return false;
  }
  if (lib == ERR_LIB_EVP &&
      (reason == EVP_R_UNKNOWN_MESSAGE_DIGEST_ALGORITHM ||
       reason == EVP_R_UNKNOWN_SIGNATURE_ALGORITHM)) {
    return LogUnsupportedAlgorithm();
  }
#else
  // OpenSSL returns 0 for simple verification failures, and -1 for
  // "exceptional circumstances".
  if (ret == 0) {
    ClearOpenSSLErrors();
    return false;
  }
  if (lib == ERR_LIB_ASN1 &&
      (reason == ASN1_R_UNKNOWN_MESSAGE_DIGEST_ALGORITHM ||
       reason == ASN1_R_UNKNOWN_SIGNATURE_ALGORITHM)) {
    return LogUnsupportedAlgorithm();
  }
#endif
  LOG(ERROR) << "OpenSSL X509_verify returned " << ret;
  LOG_OPENSSL_ERRORS(ERROR);
  return util::Status(Code::INTERNAL, "X509 verify error");
}


util::Status Cert::DerEncoding(string* result) const {
  if (!IsLoaded()) {
    LOG(ERROR) << "Cert not loaded";
    return util::Status(Code::FAILED_PRECONDITION, "Cert not loaded");
  }

  unsigned char* der_buf(nullptr);
  int der_length = i2d_X509(x509_.get(), &der_buf);

  if (der_length < 0) {
    // What does this return value mean? Let's assume it means the cert
    // is bad until proven otherwise.
    LOG(WARNING) << "Failed to serialize cert";
    LOG_OPENSSL_ERRORS(WARNING);
    return util::Status(Code::INVALID_ARGUMENT, "DER decoding failed");
  }

  result->assign(reinterpret_cast<char*>(der_buf), der_length);
  OPENSSL_free(der_buf);
  return util::Status::OK;
}


util::Status Cert::PemEncoding(string* result) const {
  if (!IsLoaded()) {
    LOG(ERROR) << "Cert not loaded";
    return util::Status(Code::FAILED_PRECONDITION, "Cert not loaded");
  }

  ScopedBIO bp(BIO_new(BIO_s_mem()));
  if (!PEM_write_bio_X509(bp.get(), x509_.get())) {
    LOG(WARNING) << "Failed to serialize cert";
    LOG_OPENSSL_ERRORS(WARNING);
    return util::Status(Code::INVALID_ARGUMENT, "PEM serialize failed");
  }

  char* data;
  const long len(BIO_get_mem_data(bp.get(), &data));
  CHECK_GT(len, 0);
  CHECK(data);

  result->assign(data, len);

  return util::Status::OK;
}


util::Status Cert::Sha256Digest(string* result) const {
  if (!IsLoaded()) {
    LOG(ERROR) << "Cert not loaded";
    return util::Status(Code::FAILED_PRECONDITION, "Cert not loaded");
  }

  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned int len;
  if (X509_digest(x509_.get(), EVP_sha256(), digest, &len) != 1) {
    // What does this return value mean? Let's assume it means the cert
    // is bad until proven otherwise.
    LOG(WARNING) << "Failed to compute cert digest";
    LOG_OPENSSL_ERRORS(WARNING);
    return util::Status(Code::INVALID_ARGUMENT, "SHA256 digest failed");
  }

  result->assign(reinterpret_cast<char*>(digest), len);
  return util::Status::OK;
}


util::Status Cert::DerEncodedTbsCertificate(string* result) const {
  if (!IsLoaded()) {
    LOG(ERROR) << "Cert not loaded";
    return util::Status(Code::FAILED_PRECONDITION, "Cert not loaded");
  }

  unsigned char* der_buf(nullptr);
  int der_length = i2d_re_X509_tbs(x509_.get(), &der_buf);
  if (der_length < 0) {
    // What does this return value mean? Let's assume it means the cert
    // is bad until proven otherwise.
    LOG(WARNING) << "Failed to serialize the TBS component";
    LOG_OPENSSL_ERRORS(WARNING);
    return util::Status(Code::INVALID_ARGUMENT, "TBS DER serialize failed");
  }
  result->assign(reinterpret_cast<char*>(der_buf), der_length);
  OPENSSL_free(der_buf);
  return util::Status::OK;
}


util::Status Cert::DerEncodedSubjectName(string* result) const {
  if (!IsLoaded()) {
    LOG(ERROR) << "Cert not loaded";
    return util::Status(Code::FAILED_PRECONDITION, "Cert not loaded");
  }
  return DerEncodedName(X509_get_subject_name(x509_.get()), result);
}


util::Status Cert::DerEncodedIssuerName(string* result) const {
  if (!IsLoaded()) {
    LOG(ERROR) << "Cert not loaded";
    return util::Status(Code::FAILED_PRECONDITION, "Cert not loaded");
  }
  return DerEncodedName(X509_get_issuer_name(x509_.get()), result);
}


// static
util::Status Cert::DerEncodedName(X509_NAME* name, string* result) {
  unsigned char* der_buf(nullptr);
  int der_length = i2d_X509_NAME(name, &der_buf);
  if (der_length < 0) {
    // What does this return value mean? Let's assume it means the cert
    // is bad until proven otherwise.
    LOG(WARNING) << "Failed to serialize the subject name";
    LOG_OPENSSL_ERRORS(WARNING);
    return util::Status(Code::INVALID_ARGUMENT, "name DER serialize failed");
  }
  result->assign(reinterpret_cast<char*>(der_buf), der_length);
  OPENSSL_free(der_buf);
  return util::Status::OK;
}


util::Status Cert::PublicKeySha256Digest(string* result) const {
  if (!IsLoaded()) {
    LOG(ERROR) << "Cert not loaded";
    return util::Status(Code::FAILED_PRECONDITION, "Cert not loaded");
  }

  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned int len;
  if (X509_pubkey_digest(x509_.get(), EVP_sha256(), digest, &len) != 1) {
    // What does this return value mean? Let's assume it means the cert
    // is bad until proven otherwise.
    LOG(WARNING) << "Failed to compute public key digest";
    LOG_OPENSSL_ERRORS(WARNING);
    return util::Status(Code::INVALID_ARGUMENT, "SHA256 digest failed");
  }
  result->assign(reinterpret_cast<char*>(digest), len);
  return util::Status::OK;
}


util::Status Cert::SPKISha256Digest(string* result) const {
  if (!IsLoaded()) {
    LOG(ERROR) << "Cert not loaded";
    return util::Status(Code::FAILED_PRECONDITION, "Cert not loaded");
  }

  unsigned char* der_buf(nullptr);
  int der_length =
      i2d_X509_PUBKEY(X509_get_X509_PUBKEY(x509_.get()), &der_buf);
  if (der_length < 0) {
    // What does this return value mean? Let's assume it means the cert
    // is bad until proven otherwise.
    LOG(WARNING) << "Failed to serialize the Subject Public Key Info";
    LOG_OPENSSL_ERRORS(WARNING);
    return util::Status(Code::INVALID_ARGUMENT, "SPKI SHA256 digest failed");
  }

  string sha256_digest = Sha256Hasher::Sha256Digest(
      string(reinterpret_cast<char*>(der_buf), der_length));

  result->assign(sha256_digest);
  OPENSSL_free(der_buf);
  return util::Status::OK;
}

util::Status Cert::OctetStringExtensionData(int extension_nid,
                                            string* result) const {
  if (!IsLoaded()) {
    LOG(ERROR) << "Cert not loaded";
    return util::Status(Code::FAILED_PRECONDITION, "Cert not loaded");
  }

  // Callers don't care whether extension is missing or invalid as they
  // usually call this method after confirming it to be present.
  const StatusOr<void*> ext_struct = ExtensionStructure(extension_nid);
  if (!ext_struct.ok() &&
      ext_struct.status().CanonicalCode() == Code::NOT_FOUND) {
    return ext_struct.status();
  }

  // |octet| is never null upon success. Caller is responsible for the
  // correctness of this cast.
  ScopedASN1_OCTET_STRING octet(
      static_cast<ASN1_OCTET_STRING*>(ext_struct.ValueOrDie()));
  result->assign(reinterpret_cast<const char*>(octet->data), octet->length);
  return util::Status::OK;
}


util::StatusOr<int> Cert::ExtensionIndex(int extension_nid) const {
  const int index(X509_get_ext_by_NID(x509_.get(), extension_nid, -1));
  if (index < -1) {
    // The most likely and possibly only cause for a return code
    // other than -1 is an unrecognized NID.
    LOG(ERROR) << "OpenSSL X509_get_ext_by_NID returned " << index
               << " for NID " << extension_nid
               << ". Is the NID not recognised?";
    LOG_OPENSSL_ERRORS(ERROR);
    return util::Status(util::error::INTERNAL, "X509_get_ext_by_NID error");
  }
  if (index == -1)
    return util::Status(util::error::NOT_FOUND, "extension not found");
  return index;
}


StatusOr<X509_EXTENSION*> Cert::GetExtension(int extension_nid) const {
  const StatusOr<int> extension_index(ExtensionIndex(extension_nid));
  if (!extension_index.ok()) {
    return extension_index.status();
  }

  X509_EXTENSION* const ext(
      X509_get_ext(x509_.get(), extension_index.ValueOrDie()));
  if (!ext) {
    LOG(ERROR) << "Failed to retrieve extension for NID " << extension_nid
               << ", at index " << extension_index.ValueOrDie();
    LOG_OPENSSL_ERRORS(ERROR);
    return util::Status(util::error::INTERNAL,
                        "failed to retrieve extension for NID " +
                            to_string(extension_nid) + ", at index " +
                            to_string(extension_index.ValueOrDie()));
  }

  return ext;
}


util::StatusOr<void*> Cert::ExtensionStructure(int extension_nid) const {
  // Let's first check if the extension is present. This allows us to
  // distinguish between "NID not recognized" and the more harmless
  // "extension not found, found more than once or corrupt".
  const StatusOr<bool> has_ext = HasExtension(extension_nid);
  if (!has_ext.ok()) {
    return has_ext.status();
  }

  if (!has_ext.ValueOrDie()) {
    return util::Status(Code::NOT_FOUND, "Extension NID " +
                                             to_string(extension_nid) +
                                             " not present or invalid");
  }

  int crit;

  void* ext_struct(
      X509_get_ext_d2i(x509_.get(), extension_nid, &crit, nullptr));

  if (!ext_struct) {
    if (crit != -1) {
      LOG(WARNING) << "Corrupt extension data";
      LOG_OPENSSL_ERRORS(WARNING);
    }

    return util::Status(Code::FAILED_PRECONDITION,
                        "Corrupt extension in cert?");
  }

  return ext_struct;
}


bool IsRedactedHost(const string& hostname) {
  // Split the hostname on '.' characters
  const vector<string> tokens(util::split(hostname, '.'));

  for (const string& str : tokens) {
    if (str == "?") {
      return true;
    }
  }

  return false;
}


bool IsValidRedactedHost(const string& hostname) {
  // Split the hostname on '.' characters
  const vector<string> tokens(util::split(hostname, '.'));

  // Enforces the following rules: '?' must be to left of non redactions
  // If first label is '*' then treat it as if it was a redaction
  bool can_redact = true;
  for (size_t pos = 0; pos < tokens.size(); ++pos) {
    if (tokens[pos] == "?") {
      if (!can_redact) {
        return false;
      }
    } else {
      // Allow a leading '*' for redaction but once we've seen anything else
      // forbid further redactions
      if (tokens[pos] != "*") {
        can_redact = false;
      } else if (pos > 0) {
        // '*' is only valid at the left
        return false;
      }
    }
  }

  return true;
}


namespace {


bool ValidateRedactionSubjectAltNames(STACK_OF(GENERAL_NAME) *
                                          subject_alt_names,
                                      vector<string>* dns_alt_names,
                                      util::Status* status,
                                      int* redacted_name_count) {
  // First. Check all the Subject Alt Name extension records. Any that are of
  // type DNS must pass validation if they are attempting to redact labels
  if (subject_alt_names) {
    const int subject_alt_name_count = sk_GENERAL_NAME_num(subject_alt_names);

    for (int i = 0; i < subject_alt_name_count; ++i) {
      GENERAL_NAME* const name(sk_GENERAL_NAME_value(subject_alt_names, i));

      util::Status name_status;

      if (name->type == GEN_DNS) {
        const string dns_name =
            ASN1ToStringAndCheckForNulls(name->d.dNSName, "DNS name",
                                         &name_status);

        if (!name_status.ok()) {
          *status = name_status;
          return true;
        }

        dns_alt_names->push_back(dns_name);

        if (IsRedactedHost(dns_name)) {
          if (!IsValidRedactedHost(dns_name)) {
            LOG(WARNING) << "Invalid redacted host: " << dns_name;
            *status = util::Status(Code::INVALID_ARGUMENT,
                                   "Invalid redacted hostname");
            return true;
          }

          redacted_name_count++;
        }
      }
    }
  }

  // This stage of validation is complete, result is not final yet
  return false;
}


}  // namespace


// Helper method for validating V2 redaction rules. If it returns true
// then the result in status is final.
bool Cert::ValidateRedactionSubjectAltNameAndCN(int* dns_alt_name_count,
                                                util::Status* status) const {
  string common_name;
  int redacted_name_count = 0;
  vector<string> dns_alt_names;

  ScopedGENERAL_NAMEStack subject_alt_names(
      static_cast<STACK_OF(GENERAL_NAME)*>(X509_get_ext_d2i(
          x509_.get(), NID_subject_alt_name, nullptr, nullptr)));

  // Apply validation rules for subject alt names, if this returns true
  // status is already final.
  if (subject_alt_names &&
      ValidateRedactionSubjectAltNames(subject_alt_names.get(), &dns_alt_names,
                                       status, &redacted_name_count)) {
    return true;
  }

  // The next stage of validation is that if the subject name CN exists it
  // must match the first DNS id and have the same labels redacted
  // TODO: Confirm it's valid to not have a CN.
  X509_NAME* const name(X509_get_subject_name(x509_.get()));

  if (!name) {
    LOG(ERROR) << "Missing X509 subject name";
    *status =
        util::Status(Code::INVALID_ARGUMENT, "Missing X509 subject name");
    return true;
  }

  const int name_pos(X509_NAME_get_index_by_NID(name, NID_commonName, -1));

  if (name_pos >= 0) {
    X509_NAME_ENTRY* const name_entry(X509_NAME_get_entry(name, name_pos));

    if (name_entry) {
      ASN1_STRING* const subject_name_asn1(
          X509_NAME_ENTRY_get_data(name_entry));

      if (!subject_name_asn1) {
        LOG(WARNING) << "Missing subject name";
        // TODO: Check this is correct behaviour. Is it OK to not have
        // a subject?
      } else {
        util::Status cn_status;
        common_name =
            ASN1ToStringAndCheckForNulls(subject_name_asn1, "CN", &cn_status);

        if (!cn_status.ok()) {
          *status = cn_status;
          return true;
        }
      }
    }
  }

  // If both a subject CN and DNS ids are present in the cert then the
  // first DNS id must exactly match the CN
  if (!dns_alt_names.empty() && !common_name.empty()) {
    if (dns_alt_names[0] != common_name) {
      LOG(WARNING) << "CN " << common_name << " does not match DNS.0 "
                   << dns_alt_names[0];
      *status =
          util::Status(Code::INVALID_ARGUMENT, "CN does not match DNS.0");
      return true;
    }
  }

  // The attempted redaction passes host validation. Stage two is checking
  // that the required extensions are present and specified correctly if
  // we found any redacted names. First though if nothing is redacted
  // then the rest of the rules need not be applied
  if (redacted_name_count == 0 && !IsRedactedHost(common_name)) {
    *status = util::Status::OK;
    return true;
  }

  *dns_alt_name_count = dns_alt_names.size();
  return false;  // validation has no definite result yet
}


util::Status Cert::IsValidWildcardRedaction() const {
  if (!IsLoaded()) {
    LOG(ERROR) << "Cert not loaded";
    return util::Status(Code::FAILED_PRECONDITION, "Cert not loaded");
  }

  util::Status status(Code::UNKNOWN, "Unknown error");
  int dns_alt_name_count = 0;

  // First we apply all the checks to the subject CN and the list of DNS
  // names in subject alt names. If these checks have a definite result
  // then return it immediately.
  if (ValidateRedactionSubjectAltNameAndCN(&dns_alt_name_count, &status)) {
    return status;
  }

  // If we reach here then the RFC says the CT redaction count extension
  // MUST BE present.
  const StatusOr<X509_EXTENSION*> exty(
      GetExtension(NID_ctPrecertificateRedactedLabelCount));
  if (!exty.ok()) {
    LOG(WARNING)
        << "required CT redaction count extension could not be found in cert";
    return util::Status(Code::INVALID_ARGUMENT,
                        "No CT redaction count extension");
  }

  // Ensure the data in the extension is a sequence. DER encoding is same for
  // SEQUENCE and SEQUENCE OF and we'll check types later.
  if (exty.ValueOrDie()->value->data[0] !=
      (V_ASN1_SEQUENCE | V_ASN1_CONSTRUCTED)) {
    LOG(WARNING) << "CT redaction count extension is not a SEQUENCE OF";
    return util::Status(Code::INVALID_ARGUMENT,
                        "CT redaction count extension not a sequence");
  }

  // Unpack the extension contents, which should be SEQUENCE OF INTEGER.
  // For compatibility we unpack any sequence and check integer type as we go.
  // Don't pass the pointer from the extension directly as it gets incremented
  // during parsing.
  const unsigned char* sequence_data(
      const_cast<const unsigned char*>(exty.ValueOrDie()->value->data));
  ScopedASN1_TYPEStack asn1_types(static_cast<STACK_OF(ASN1_TYPE)*>(
      d2i_ASN1_SEQUENCE_ANY(nullptr, &sequence_data,
                            exty.ValueOrDie()->value->length)));

  if (asn1_types) {
    const int num_integers(sk_ASN1_TYPE_num(asn1_types.get()));

    // RFC text says there MUST NOT be more integers than there are DNS ids
    if (num_integers > dns_alt_name_count) {
      LOG(WARNING) << "Too many integers in extension: " << num_integers
                   << " but only " << dns_alt_name_count << " DNS names";
      return util::Status(Code::INVALID_ARGUMENT,
                          "More integers in ext than redacted labels");
    }

    // All the integers in the sequence must be positive, check the sign
    // after conversion to BIGNUM
    for (int i = 0; i < num_integers; ++i) {
      ASN1_TYPE* const asn1_type(sk_ASN1_TYPE_value(asn1_types.get(), i));

      if (asn1_type->type != V_ASN1_INTEGER) {
        LOG(WARNING) << "Redaction count has non-integer in sequence"
                     << asn1_type->type;
        return util::Status(Code::INVALID_ARGUMENT,
                            "Non integer found in redaction label count");
      }

      ASN1_INTEGER* const redacted_labels(asn1_type->value.integer);
      ScopedBIGNUM value(ASN1_INTEGER_to_BN(redacted_labels, nullptr));

      const bool neg = value->neg;
      if (neg) {
        ScopedOpenSSLString bn_hex(BN_bn2hex(value.get()));
        LOG(WARNING) << "Invalid negative redaction label count: "
                     << bn_hex.get();
        return util::Status(Code::INVALID_ARGUMENT, "Invalid -ve label count");
      }
    }

  } else {
    LOG(WARNING) << "Failed to unpack SEQUENCE OF in CT extension";
    return util::Status(Code::INVALID_ARGUMENT,
                        "Failed to unpack integer sequence in ext");
  }

  return util::Status::OK;
}


util::Status Cert::IsValidNameConstrainedIntermediateCa() const {
  if (!IsLoaded()) {
    LOG(ERROR) << "Cert not loaded";
    return util::Status(Code::FAILED_PRECONDITION, "Cert not loaded");
  }

  // If it's not a CA cert or there is no name constraint extension then we
  // don't need to apply the rules any further
  const StatusOr<bool> has_ca_constraint = HasBasicConstraintCATrue();
  const StatusOr<bool> has_name_constraints =
      HasExtension(NID_name_constraints);

  // However, we don't expect either of the above lookups to fail as the
  // extensions are registered.
  if (!has_ca_constraint.ok()) {
    return has_ca_constraint.status();
  }

  if (!has_name_constraints.ok()) {
    return has_name_constraints.status();
  }

  if (!has_ca_constraint.ValueOrDie() || !has_name_constraints.ValueOrDie()) {
    return util::Status::OK;
  }

  // So there now must be a CT extension and the name constraint must not be
  // in error
  const StatusOr<bool> has_ct_nolog_intermediate =
      HasExtension(NID_ctNameConstraintNologIntermediateCa);

  CHECK(has_name_constraints.ValueOrDie());
  if (!has_ct_nolog_intermediate.ok() ||
      !has_ct_nolog_intermediate.ValueOrDie()) {
    LOG(WARNING) << "Name constraint extension without CT extension";
    return util::Status(Code::INVALID_ARGUMENT,
                        "Name constraint ext present, CT ext missing");
  }

  int crit;
  NAME_CONSTRAINTS* const nc(static_cast<NAME_CONSTRAINTS*>(
      X509_get_ext_d2i(x509_.get(), NID_name_constraints, &crit, nullptr)));

  if (!nc || crit == -1) {
    LOG(ERROR) << "Couldn't parse the name constraint extension";
    return util::Status(Code::INTERNAL, "Failed to parse name constraint");
  }

  // Search all the permitted subtrees, there must be at least one DNS
  // entry and it must not be empty
  bool seen_dns = false;

  for (int permitted_subtree = 0;
       permitted_subtree < sk_GENERAL_SUBTREE_num(nc->permittedSubtrees);
       ++permitted_subtree) {
    GENERAL_SUBTREE* const perm_subtree(
        sk_GENERAL_SUBTREE_value(nc->permittedSubtrees, permitted_subtree));

    if (perm_subtree->base && perm_subtree->base->type == GEN_DNS &&
        perm_subtree->base->d.dNSName->length > 0) {
      seen_dns = true;
    }
  }

  // There must be an excluded subtree entry that covers the whole IPv4 and
  // IPv6 range. Or at least one entry for both that covers the whole
  // range
  bool seen_ipv4 = false;
  bool seen_ipv6 = false;

  // TODO: Does not handle more complex cases at the moment and I'm
  // not sure whether it should. E.g. a combination of multiple entries
  // that end up covering the whole available range. For the moment
  // things similar to the example in the RFC work.
  for (int excluded_subtree = 0;
       excluded_subtree < sk_GENERAL_SUBTREE_num(nc->excludedSubtrees);
       ++excluded_subtree) {
    GENERAL_SUBTREE* const excl_subtree(
        sk_GENERAL_SUBTREE_value(nc->excludedSubtrees, excluded_subtree));

    // Only consider entries that are of type ipAddress (OCTET_STRING)
    if (excl_subtree->base && excl_subtree->base->type == GEN_IPADD) {
      // First check that all the bytes of the string are zero
      bool all_zero = true;
      for (int i = 0; i < excl_subtree->base->d.ip->length; ++i) {
        if (excl_subtree->base->d.ip->data[i] != 0) {
          all_zero = false;
        }
      }

      if (all_zero) {
        if (excl_subtree->base->d.ip->length == 32) {
          // IPv6
          seen_ipv6 = true;
        } else if (excl_subtree->base->d.ip->length == 8) {
          // IPv4
          seen_ipv4 = true;
        }
      }
    }
  }

  NAME_CONSTRAINTS_free(nc);

  if (!seen_dns) {
    LOG(WARNING) << "No DNS entry found in permitted subtrees";
    return util::Status(Code::INVALID_ARGUMENT,
                        "No DNS entry in permitted subtrees");
  }

  if (!seen_ipv4 || !seen_ipv6) {
    LOG(WARNING) << "Excluded subtree does not cover all IPv4 and v6 range";
    return util::Status(Code::INVALID_ARGUMENT,
                        "Does not exclude all IPv4 and v6 range");
  }

  return util::Status::OK;
}


TbsCertificate::TbsCertificate(const Cert& cert) {
  if (!cert.IsLoaded()) {
    LOG(ERROR) << "Cert not loaded";
    return;
  }

  x509_.reset(X509_dup(cert.x509_.get()));

  if (!x509_)
    LOG_OPENSSL_ERRORS(ERROR);
}


util::Status TbsCertificate::DerEncoding(string* result) const {
  if (!IsLoaded()) {
    LOG(ERROR) << "TBS not loaded";
    return util::Status(Code::FAILED_PRECONDITION, "Cert not loaded (TBS)");
  }

  unsigned char* der_buf(nullptr);
  int der_length = i2d_re_X509_tbs(x509_.get(), &der_buf);
  if (der_length < 0) {
    // What does this return value mean? Let's assume it means the cert
    // is bad until proven otherwise.
    LOG(WARNING) << "Failed to serialize the TBS component";
    LOG_OPENSSL_ERRORS(WARNING);
    return util::Status(Code::INTERNAL, "Failed to serialize TBS");
  }
  result->assign(reinterpret_cast<char*>(der_buf), der_length);
  OPENSSL_free(der_buf);
  return util::Status::OK;
}


util::Status TbsCertificate::DeleteExtension(int extension_nid) {
  if (!IsLoaded()) {
    LOG(ERROR) << "TBS not loaded";
    return util::Status(Code::FAILED_PRECONDITION, "Cert not loaded (TBS)");
  }

  const StatusOr<int> extension_index(ExtensionIndex(extension_nid));
  // If the extension doesn't exist then there is nothing to do and this
  // propagates the NOT_FOUND status.
  if (!extension_index.ok()) {
    return extension_index.status();
  }

  ScopedX509_EXTENSION ext(
      X509_delete_ext(x509_.get(), extension_index.ValueOrDie()));

  if (!ext) {
    // Truly odd.
    LOG(ERROR) << "Failed to delete the extension";
    LOG_OPENSSL_ERRORS(ERROR);
    return util::Status(Code::INTERNAL, "Failed to delete extension");
  }


  // ExtensionIndex returns the first matching index - if the extension
  // occurs more than once, just give up.
  const StatusOr<int> ignored_index(ExtensionIndex(extension_nid));
  if (ignored_index.ok()) {
    LOG(WARNING)
        << "Failed to delete the extension. Does the certificate have "
        << "duplicate extensions?";
    return util::Status(Code::ALREADY_EXISTS, "Multiple extensions in cert");
  }

  // It's not an error if the extension didn't exist the second time
  // as it should have been deleted.
  if (!ignored_index.ok() &&
      ignored_index.status().CanonicalCode() != Code::NOT_FOUND) {
    return ignored_index.status();
  }

  return util::Status::OK;
}


util::Status TbsCertificate::CopyIssuerFrom(const Cert& from) {
  if (!from.IsLoaded()) {
    LOG(ERROR) << "Cert not loaded";
    return util::Status(Code::FAILED_PRECONDITION, "Cert not loaded");
  }

  if (!IsLoaded()) {
    LOG(ERROR) << "TBS not loaded";
    return util::Status(Code::FAILED_PRECONDITION, "Cert not loaded (TBS)");
  }

  // This just looks up the relevant pointer so there shouldn't
  // be any errors to clear.
  X509_NAME* ca_name = X509_get_issuer_name(from.x509_.get());
  if (!ca_name) {
    LOG(WARNING) << "Issuer certificate has NULL name";
    return util::Status(Code::FAILED_PRECONDITION,
                        "Issuer cert has NULL name");
  }

  if (X509_set_issuer_name(x509_.get(), ca_name) != 1) {
    LOG(WARNING) << "Failed to set issuer name, Cert has NULL issuer?";
    LOG_OPENSSL_ERRORS(WARNING);
    return util::Status(Code::FAILED_PRECONDITION,
                        "Failed to set issuer, possibly null?");
  }

  // Verify that the Authority KeyID extensions are compatible.
  StatusOr<int> status = ExtensionIndex(NID_authority_key_identifier);
  if (status.status().CanonicalCode() == Code::NOT_FOUND) {
    // No extension found = nothing to copy
    return util::Status::OK;
  }

  if (!status.ok() || !status.ValueOrDie()) {
    LOG(ERROR) << "Failed to check Authority Key Identifier extension";
    return util::Status(Code::INTERNAL,
                        "Failed to check Authority KeyID extension (TBS)");
  }

  const StatusOr<int> from_extension_index(
      from.ExtensionIndex(NID_authority_key_identifier));
  if (from_extension_index.status().CanonicalCode() ==
      util::error::NOT_FOUND) {
    // No extension found = cannot copy.
    LOG(WARNING) << "Unable to copy issuer: destination has an Authority "
                 << "KeyID extension, but the source has none.";
    return util::Status(Code::FAILED_PRECONDITION,
                        "Incompatible Authority KeyID extensions");
  }

  if (!from_extension_index.ok()) {
    LOG(ERROR) << "Failed to check Authority Key Identifier extension";
    return util::Status(Code::INTERNAL,
                        "Failed to check Authority KeyID extension");
  }

  // Ok, now copy the extension, keeping the critical bit (which should always
  // be false in a valid cert, mind you).
  X509_EXTENSION* to_ext = X509_get_ext(x509_.get(), status.ValueOrDie());
  X509_EXTENSION* from_ext =
      X509_get_ext(from.x509_.get(), from_extension_index.ValueOrDie());

  if (!to_ext || !from_ext) {
    // Should not happen.
    LOG(ERROR) << "Failed to retrieve extension";
    LOG_OPENSSL_ERRORS(ERROR);
    return util::Status(Code::INTERNAL,
                        "Failed to retrieve one or both extensions");
  }

  if (X509_EXTENSION_set_data(to_ext, X509_EXTENSION_get_data(from_ext)) !=
      1) {
    LOG(ERROR) << "Failed to copy extension data.";
    LOG_OPENSSL_ERRORS(ERROR);
    return util::Status(Code::INTERNAL, "Failed to copy extension data");
  }

  return util::Status::OK;
}


StatusOr<int> TbsCertificate::ExtensionIndex(int extension_nid) const {
  int index = X509_get_ext_by_NID(x509_.get(), extension_nid, -1);
  if (index < -1) {
    // The most likely and possibly only cause for a return code
    // other than -1 is an unrecognized NID. This is different from a
    // known extension not being present.
    LOG(ERROR) << "OpenSSL X509_get_ext_by_NID returned " << index
               << " for NID " << extension_nid
               << ". Is the NID not recognised?";
    LOG_OPENSSL_ERRORS(ERROR);
    return util::Status(Code::INTERNAL,
                        "Extension lookup failed. Incorrect NID?");
  }
  if (index == -1) {
    return util::Status(Code::NOT_FOUND, "Extension not found.");
  }

  return index;
}


CertChain::CertChain(const string& pem_string) {
  // A read-only BIO.
  ScopedBIO bio_in(BIO_new_mem_buf(const_cast<char*>(pem_string.data()),
                                   pem_string.length()));
  if (!bio_in) {
    LOG_OPENSSL_ERRORS(ERROR);
    return;
  }

  X509* x509(nullptr);
  while ((x509 = PEM_read_bio_X509(bio_in.get(), nullptr, nullptr, nullptr))) {
    chain_.push_back(new Cert(x509));
  }

  // The last error must be EOF.
  unsigned long err = ERR_peek_last_error();
  if (ERR_GET_LIB(err) != ERR_LIB_PEM ||
      ERR_GET_REASON(err) != PEM_R_NO_START_LINE) {
    // A real error.
    LOG(WARNING) << "Input is not a valid PEM-encoded certificate chain";
    LOG_OPENSSL_ERRORS(WARNING);
    ClearChain();
  } else {
    ClearOpenSSLErrors();
  }
}


bool CertChain::AddCert(Cert* cert) {
  if (!cert || !cert->IsLoaded()) {
    LOG(ERROR) << "Attempting to add an invalid cert";
    if (cert)
      delete cert;
    return false;
  }
  chain_.push_back(cert);
  return true;
}


void CertChain::RemoveCert() {
  if (IsLoaded()) {
    delete chain_.back();
    chain_.pop_back();
  } else {
    LOG(ERROR) << "Chain is not loaded";
  }
}


bool CertChain::RemoveCertsAfterFirstSelfSigned() {
  if (!IsLoaded()) {
    LOG(ERROR) << "Chain is not loaded";
    return false;
  }

  size_t first_self_signed = chain_.size();

  // Find the first self-signed certificate.
  for (size_t i = 0; i < chain_.size(); ++i) {
    StatusOr<bool> status = chain_[i]->IsSelfSigned();
    if (!status.ok()) {
      return false;
    } else if (status.ValueOrDie()) {
      first_self_signed = i;
      break;
    }
  }

  if (first_self_signed == chain_.size())
    return true;

  // Remove everything after it.
  size_t chain_size = chain_.size();
  for (size_t i = first_self_signed + 1; i < chain_size; ++i) {
    RemoveCert();
  }
  return true;
}


CertChain::~CertChain() {
  ClearChain();
}


util::Status CertChain::IsValidCaIssuerChainMaybeLegacyRoot() const {
  if (!IsLoaded()) {
    LOG(ERROR) << "Chain is not loaded";
    return util::Status(Code::FAILED_PRECONDITION, "Cert not loaded");
  }

  for (vector<Cert*>::const_iterator it = chain_.begin();
       it + 1 < chain_.end(); ++it) {
    Cert* subject = *it;
    Cert* issuer = *(it + 1);

    // The root cert may not have CA:True
    const StatusOr<bool> status = issuer->IsSelfSigned();
    if (status.ok() && !status.ValueOrDie()) {
      const StatusOr<bool> s2(issuer->HasBasicConstraintCATrue());
      if (!s2.ok() || !s2.ValueOrDie()) {
        return util::Status(Code::INVALID_ARGUMENT,
                            "CA constraint check failed");
      }
    } else if (!status.ok()) {
      LOG(ERROR) << "Failed to check self-signed status";
      return util::Status(Code::INVALID_ARGUMENT,
                          "Failed to check self signed status");
    }

    const StatusOr<bool> s3 = subject->IsIssuedBy(*issuer);
    if (!s3.ok() || !s3.ValueOrDie()) {
      return util::Status(Code::INVALID_ARGUMENT, "Issuer check failed");
    }
  }
  return util::Status::OK;
}


util::Status CertChain::IsValidSignatureChain() const {
  if (!IsLoaded()) {
    LOG(ERROR) << "Chain is not loaded";
    return util::Status(util::error::FAILED_PRECONDITION,
                        "certificate chain is not loaded");
  }

  for (vector<Cert*>::const_iterator it = chain_.begin();
       it + 1 < chain_.end(); ++it) {
    Cert* subject = *it;
    Cert* issuer = *(it + 1);

    const StatusOr<bool> status = subject->IsSignedBy(*issuer);

    // Propagate any failure status if we get one. This includes
    // UNIMPLEMENTED for unsupported algorithms. This can happen
    // when a weak algorithm (such as MD2) is intentionally not
    // accepted in which case it's correct to say that the chain is invalid.
    // It can also happen when EVP is not properly initialized, in
    // which case it's more of an INTERNAL_ERROR. However a bust
    // setup would manifest itself in many other ways, including
    // failing tests, so we assume the failure is intentional.
    if (!status.ok()) {
      return status.status();
    }

    // Must have been signed by issuer or it's an invalid chain
    if (!status.ValueOrDie()) {
      return util::Status(util::error::INVALID_ARGUMENT,
                          "invalid certificate chain");
    }
  }

  return util::Status::OK;
}


void CertChain::ClearChain() {
  vector<Cert*>::const_iterator it;
  for (it = chain_.begin(); it < chain_.end(); ++it)
    delete *it;
  chain_.clear();
}


util::StatusOr<bool> PreCertChain::UsesPrecertSigningCertificate() const {
  const Cert* issuer = PrecertIssuingCert();
  if (!issuer) {
    // No issuer, so it must be a real root CA from the store.
    return false;
  }

  return issuer->HasExtendedKeyUsage(cert_trans::NID_ctPrecertificateSigning);
}


util::StatusOr<bool> PreCertChain::IsWellFormed() const {
  if (!IsLoaded()) {
    LOG(ERROR) << "Chain is not loaded";
    return util::Status(Code::FAILED_PRECONDITION, "Cert not loaded");
  }

  const Cert* pre = PreCert();

  // (1) Check that the leaf contains the critical poison extension.
  const StatusOr<bool> has_poison =
      pre->HasCriticalExtension(cert_trans::NID_ctPoison);
  if (!has_poison.ok() || !has_poison.ValueOrDie()) {
    return has_poison;
  }

  // (2) If signed by a Precertificate Signing Certificate, check that
  // the AKID extensions are compatible.
  const StatusOr<bool> uses_precert_signing = UsesPrecertSigningCertificate();
  if (uses_precert_signing.ok() && !uses_precert_signing.ValueOrDie()) {
    // If there is no precert signing extendedKeyUsage, no more checks:
    // the cert was issued by a regular CA.
    return true;
  }

  if (!uses_precert_signing.ok()) {
    return uses_precert_signing.status();
  }

  CHECK(uses_precert_signing.ValueOrDie());

  const Cert* issuer = PrecertIssuingCert();
  // If pre has the extension set but the issuer doesn't, error.
  const StatusOr<bool> has_akid =
      pre->HasExtension(NID_authority_key_identifier);

  if (has_akid.ok() && !has_akid.ValueOrDie()) {
    return true;
  }
  if (!has_akid.ok()) {
    return has_akid;
  }

  CHECK(has_akid.ValueOrDie());

  // Extension present in the leaf: check it's present in the issuer.
  return issuer->HasExtension(NID_authority_key_identifier);
}


}  // namespace cert_trans
