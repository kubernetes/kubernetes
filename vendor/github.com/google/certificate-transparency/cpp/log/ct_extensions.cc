#include "log/ct_extensions.h"

#include <glog/logging.h>
#include <openssl/asn1.h>
#include <openssl/asn1t.h>
#include <openssl/objects.h>
#include <openssl/x509v3.h>
#include <string.h>

namespace cert_trans {

int NID_ctSignedCertificateTimestampList = 0;
int NID_ctEmbeddedSignedCertificateTimestampList = 0;
int NID_ctPoison = 0;
int NID_ctPrecertificateSigning = 0;
int NID_ctPrecertificateRedactedLabelCount = 0;
int NID_ctNameConstraintNologIntermediateCa = 0;
int NID_ctV2CmsPayloadContentType = 0;

// The SCT list in the extension of a superfluous certificate
const char kSCTListOID[] = "1.3.6.1.4.1.11129.2.4.1";
// The SCT list embedded in the certificate itself
const char kEmbeddedSCTListOID[] = "1.3.6.1.4.1.11129.2.4.2";
// Extension indicating consent that certs from an intermediate CA with name
// constraints may not be logged (not used in V1)
const char kNameConstraintNologIntermediateOID[] = "1.3.6.1.4.1.11129.2.4.7";
// The poison extension
const char kPoisonOID[] = "1.3.6.1.4.1.11129.2.4.3";
// Extension for wildcard redacted Precertificate indicating redaction count
// (not used in V1)
const char kPrecertificateRedactedLabelOID[] = "1.3.6.1.4.1.11129.2.4.6";
// Extended Key Usage value for Precertificate signing
const char kPrecertificateSigningOID[] = "1.3.6.1.4.1.11129.2.4.4";
// V2 Precert content type
// TODO: Required content type not defined in draft yet. Placeholder in case
// we use our own.
const char kV2PrecertificateCmsContentTypeOID[] = "1.3.6.1.4.1.11129.2.4.8";

static const char kSCTListSN[] = "ctSCT";
static const char kSCTListLN[] =
    "X509v3 Certificate Transparency Signed Certificate Timestamp List";

static const char kEmbeddedSCTListSN[] = "ctEmbeddedSCT";
static const char kEmbeddedSCTListLN[] =
    "X509v3 Certificate Transparency "
    "Embedded Signed Certificate Timestamp List";
static const char kPoisonSN[] = "ctPoison";
static const char kPoisonLN[] = "X509v3 Certificate Transparency Poison";
static const char kPrecertificateSigningSN[] = "ctPresign";
static const char kPrecertificateSigningLN[] =
    "Certificate Transparency "
    "Precertificate Signing";
static const char kPrecertificateRedactedLabelCountSN[] = "ctPreredact";
static const char kPrecertificateRedactedLabelCountLN[] =
    "Certificate Transparency "
    "Precertificate Redacted Label Count";
static const char kNameConstraintNologIntermediateSN[] =
    "ctNoLogIntermediateOk";
static const char kNameConstraintNologIntermediateLN[] =
    "Certificate Transparency "
    "Name Constrained Intermediate CA NoLog Allowed";
static const char kV2PrecertCmsContentTypeSN[] = "ctV2PrecertCmsContentType";
static const char kV2PrecertCmsContentTypeLN[] =
    "Certificate Transparency "
    "V2 Precertificate CMS Message Content Type";

static const char kASN1NullValue[] = "NULL";

// String conversion for an ASN1 NULL
static char* ASN1NullToString(X509V3_EXT_METHOD*, ASN1_NULL* asn1_null) {
  if (asn1_null == NULL)
    return NULL;
  char* buf = strdup(kASN1NullValue);
  return buf;
}

// String conversion from an ASN1:NULL conf.
static ASN1_NULL* StringToASN1Null(X509V3_EXT_METHOD*, X509V3_CTX*,
                                   char* str) {
  if (str == NULL || strcmp(str, kASN1NullValue) != 0) {
    return NULL;
  }

  return ASN1_NULL_new();
}

// clang-format off
static X509V3_EXT_METHOD ct_sctlist_method = {
    0,  // ext_nid, NID, will be created by OBJ_create()
    0,  // flags
    ASN1_ITEM_ref(ASN1_OCTET_STRING),  // the object is an octet string
    0, 0, 0, 0,                        // ignored since the field above is set
    // Create from, and print to, a hex string
    // Allows to specify the extension configuration like so:
    // ctSCT = <hexstring_value>
    // (Unused - we just plumb the bytes in the fake cert directly.)
    reinterpret_cast<X509V3_EXT_I2S>(i2s_ASN1_OCTET_STRING),
    reinterpret_cast<X509V3_EXT_S2I>(s2i_ASN1_OCTET_STRING), 0, 0, 0, 0,
    NULL  // usr_data
};

static X509V3_EXT_METHOD ct_embeddedsctlist_method = {
    0,  // ext_nid, NID, will be created by OBJ_create()
    0,  // flags
    ASN1_ITEM_ref(ASN1_OCTET_STRING),  // the object is an octet string
    0, 0, 0, 0,                        // ignored since the field above is set
    // Create from, and print to, a hex string
    // Allows to specify the extension configuration like so:
    // ctEmbeddedSCT = <hexstring_value>
    // (Unused, as we're not issuing certs.)
    reinterpret_cast<X509V3_EXT_I2S>(i2s_ASN1_OCTET_STRING),
    reinterpret_cast<X509V3_EXT_S2I>(s2i_ASN1_OCTET_STRING), 0, 0, 0, 0,
    NULL  // usr_data
};

static X509V3_EXT_METHOD ct_poison_method = {
    0,                         // ext_nid, NID, will be created by OBJ_create()
    0,                         // flags
    ASN1_ITEM_ref(ASN1_NULL),  // the object is an ASN1 NULL
    0, 0, 0, 0,                // ignored since the above is set
    // Create from, and print to, a hex string
    // Allows to specify the extension configuration like so:
    // ctPoison = "NULL"
    // (Unused, as we're not issuing certs.)
    reinterpret_cast<X509V3_EXT_I2S>(ASN1NullToString),
    reinterpret_cast<X509V3_EXT_S2I>(StringToASN1Null), 0, 0, 0, 0,
    NULL  // usr_data
};

// Not used in protocol v1. Specifies the count of redacted labels of each
// DNS id in the cert. See RFC section 3.2.2.
static X509V3_EXT_METHOD ct_redaction_count_method = {
    0,                         // ext_nid, NID, will be created by OBJ_create()
    0,                         // flags
    ASN1_ITEM_ref(REDACTED_LABEL_COUNT),
    0, 0, 0, 0,                // ignored since the above is set
    // Create from, and print to, a hex string
    // Allows to specify the extension configuration like so:
    // ctPreredact = "NULL"
    // (Unused, as we're not issuing certs.)
    reinterpret_cast<X509V3_EXT_I2S>(i2s_ASN1_OCTET_STRING),
    reinterpret_cast<X509V3_EXT_S2I>(s2i_ASN1_OCTET_STRING), 0, 0, 0, 0,
    NULL  // usr_data
};

// Not used in protocol v1. Specifies consent that name constrained
// intermediate certs may not be logged. See RFC section 3.2.3.
static X509V3_EXT_METHOD ct_name_constraint_nolog_intermediate_ca_method = {
    0,                         // ext_nid, NID, will be created by OBJ_create()
    0,                         // flags
    ASN1_ITEM_ref(ASN1_NULL),  // the object is an ASN1 NULL
    0, 0, 0, 0,                // ignored since the above is set
    // Create from, and print to, a hex string
    // Allows to specify the extension configuration like so:
    // ctPoison = "NULL"
    // (Unused, as we're not issuing certs.)
    reinterpret_cast<X509V3_EXT_I2S>(ASN1NullToString),
    reinterpret_cast<X509V3_EXT_S2I>(StringToASN1Null), 0, 0, 0, 0,
    NULL  // usr_data
};
// clang-format on

void LoadCtExtensions() {
  // V1 Certificate Extensions

  ct_sctlist_method.ext_nid = OBJ_create(kSCTListOID, kSCTListSN, kSCTListLN);
  CHECK_NE(ct_sctlist_method.ext_nid, 0);
  CHECK_EQ(1, X509V3_EXT_add(&ct_sctlist_method));
  NID_ctSignedCertificateTimestampList = ct_sctlist_method.ext_nid;

  ct_embeddedsctlist_method.ext_nid =
      OBJ_create(kEmbeddedSCTListOID, kEmbeddedSCTListSN, kEmbeddedSCTListLN);
  CHECK_NE(ct_embeddedsctlist_method.ext_nid, 0);
  CHECK_EQ(1, X509V3_EXT_add(&ct_embeddedsctlist_method));
  NID_ctEmbeddedSignedCertificateTimestampList =
      ct_embeddedsctlist_method.ext_nid;

  ct_poison_method.ext_nid = OBJ_create(kPoisonOID, kPoisonSN, kPoisonLN);
  CHECK_NE(ct_poison_method.ext_nid, 0);
  CHECK_EQ(1, X509V3_EXT_add(&ct_poison_method));
  NID_ctPoison = ct_poison_method.ext_nid;

  int precert_signing_nid =
      OBJ_create(kPrecertificateSigningOID, kPrecertificateSigningSN,
                 kPrecertificateSigningLN);
  CHECK_NE(precert_signing_nid, 0);
  NID_ctPrecertificateSigning = precert_signing_nid;

  // V2 Certificate extensions

  ct_redaction_count_method.ext_nid =
      OBJ_create(kPrecertificateRedactedLabelOID,
                 kPrecertificateRedactedLabelCountSN,
                 kPrecertificateRedactedLabelCountLN);
  CHECK_NE(ct_redaction_count_method.ext_nid, 0);
  CHECK_EQ(1, X509V3_EXT_add(&ct_redaction_count_method));
  NID_ctPrecertificateRedactedLabelCount = ct_redaction_count_method.ext_nid;

  ct_name_constraint_nolog_intermediate_ca_method.ext_nid =
      OBJ_create(kNameConstraintNologIntermediateOID,
                 kNameConstraintNologIntermediateSN,
                 kNameConstraintNologIntermediateLN);
  CHECK_NE(ct_name_constraint_nolog_intermediate_ca_method.ext_nid, 0);
  CHECK_EQ(1,
           X509V3_EXT_add(&ct_name_constraint_nolog_intermediate_ca_method));
  NID_ctNameConstraintNologIntermediateCa =
      ct_name_constraint_nolog_intermediate_ca_method.ext_nid;

  // V2 Content types

  NID_ctV2CmsPayloadContentType =
      OBJ_create(kV2PrecertificateCmsContentTypeOID,
                 kV2PrecertCmsContentTypeSN, kV2PrecertCmsContentTypeLN);
}

}  // namespace cert_trans
