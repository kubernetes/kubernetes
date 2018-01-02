#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <string>

#include "log/cert.h"
#include "log/ct_extensions.h"
#include "util/status_test_util.h"
#include "util/testing.h"
#include "util/util.h"

using cert_trans::Cert;
using cert_trans::CertChain;
using cert_trans::PreCertChain;
using cert_trans::TbsCertificate;
using std::string;
using util::error::Code;
using util::testing::StatusIs;

// TODO(ekasper): add test certs with intermediates.
// Valid certificates.
static const char kCaCert[] = "ca-cert.pem";
// Issued by ca-cert.pem
static const char kLeafCert[] = "test-cert.pem";
// Issued by ca-cert.pem
// Issued by intermediate-cert.pem
static const char kLeafWithIntermediateCert[] = "test-intermediate-cert.pem";
static const char kCaPreCert[] = "ca-pre-cert.pem";
// Issued by ca-cert.pem
static const char kPreCert[] = "test-embedded-pre-cert.pem";
// CA with no basic constraints and an MD2 signature.
static const char kLegacyCaCert[] = "test-no-bc-ca-cert.pem";

// Leaf cert with CN redaction no DNS or extension (test case 5)
static const char kV2WildcardRedactTest5[] = "redact_test5.pem";
// Leaf cert with CN/DNS wildcard redaction mismatch (test case 6)
static const char kV2WildcardRedactTest6[] = "redact_test6.pem";
// Leaf cert with valid V2 wildcard redaction (test case 7)
static const char kV2WildcardRedactTest7[] = "redact_test7.pem";
// Leaf cert with valid V2 '*' wildcard redaction (test case 8)
static const char kV2WildcardRedactTest8[] = "redact_test8.pem";
// Leaf cert with invalid redaction label (test case 9)
static const char kV2WildcardRedactTest9[] = "redact_test9.pem";
// Leaf cert with invalid V2 '*' wildcard redaction (test case 10)
static const char kV2WildcardRedactTest10[] = "redact_test10.pem";
// Leaf cert with invalid redaction, too many ext values (test case 11)
static const char kV2WildcardRedactTest11[] = "redact_test11.pem";
// Leaf cert with invalid V2 redacted label extension (test case 12)
static const char kV2WildcardRedactTest12[] = "redact_test12.pem";
// Leaf cert with not enough entries in extension (test case 13)
static const char kV2WildcardRedactTest13[] = "redact_test13.pem";
// Leaf cert with valid extension + multiple DNS entries (test case 14)
static const char kV2WildcardRedactTest14[] = "redact_test14.pem";
// Leaf cert with too many labels in extension (test case 15)
static const char kV2WildcardRedactTest15[] = "redact_test15.pem";
// Leaf cert with wildcard redaction in both CN and DNS-ID no extension
static const char kV2WildcardRedactTest22[] = "redact_test22.pem";
// Leaf cert with extension that is not a sequence (invalid)
static const char kV2WildcardRedactTest23[] = "redact_test23.pem";
// Leaf cert with extension sequence containing non integer value (invalid)
static const char kV2WildcardRedactTest24[] = "redact_test24.pem";

// Leaf cert with non CA constraints
static const char kV2ConstraintTest2[] = "constraint_test2.pem";
// Leaf cert with CA but no name constraints
static const char kV2ConstraintTest3[] = "constraint_test3.pem";
// Leaf cert with constraint and no CT ext
static const char kV2ConstraintTest4[] = "constraint_test4.pem";
// Leaf cert with constraint and CT ext but no DNS in constraint
static const char kV2ConstraintTest5[] = "constraint_test5.pem";
// Leaf cert with CA constraint + valid CT extension
static const char kV2ConstraintTest6[] = "constraint_test6.pem";
// Leaf cert with CA constraint + valid CT extension with multiple DNS
static const char kV2ConstraintTest7[] = "constraint_test7.pem";
// Leaf cert with CA constraint + valid CT extension, no IP exclude
static const char kV2ConstraintTest8[] = "constraint_test8.pem";
// Leaf cert with CA constraint + valid CT extension, partial IP exclude
static const char kV2ConstraintTest9[] = "constraint_test9.pem";

static const char kInvalidCertString[] =
    "-----BEGIN CERTIFICATE-----\ninvalid"
    "\n-----END CERTIFICATE-----\n";

static const char kMismatchingSigAlgsCertString[] =
    "-----BEGIN CERTIFICATE-----\n"
    "MIIFWjCCBD2gAwIBAgIQGERW2P+5Xt15QiwP1NRmtTANBgkqhkiG9w0BAQsFADCB\n"
    "kDELMAkGA1UEBhMCR0IxGzAZBgNVBAgTEkdyZWF0ZXIgTWFuY2hlc3RlcjEQMA4G\n"
    "A1UEBxMHU2FsZm9yZDEaMBgGA1UEChMRQ09NT0RPIENBIExpbWl0ZWQxNjA0BgNV\n"
    "BAMTLUNPTU9ETyBSU0EgRG9tYWluIFZhbGlkYXRpb24gU2VjdXJlIFNlcnZlciBD\n"
    "QTAeFw0xNDA1MjEwMDAwMDBaFw0xNTA1MjkyMzU5NTlaMFQxITAfBgNVBAsTGERv\n"
    "bWFpbiBDb250cm9sIFZhbGlkYXRlZDEUMBIGA1UECxMLUG9zaXRpdmVTU0wxGTAX\n"
    "BgNVBAMTEHZpZGVvbWFnaWNhbC5jb20wggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAw\n"
    "ggEKAoIBAQDOMDIDA7wjsHNDp+cYa9On4oodRjnAxvgJQ852ahkZul+e816WzUTT\n"
    "Bs8m0LFiXjRZtsl6AHOTVbSSK3iFrke6pVdwatVP95NsR4qQaU6ITfkD9hT01vOm\n"
    "FrPv77X7RF6C0Pb8tH9ro8prpqJdTlMnjnPTQQy/ljrUaWyIQm0G1ujCApPQhQ7h\n"
    "XRZYPAk0B5jSalA1q0tjjWKohlQaQMqXpHtbofvL9hUlWw6shJdd08tUH5o0UcW3\n"
    "so0zHvVfwi4Gw6DiMc/a8aSmNJPO09Rf+xOYGB+wyMezH900OnuhMg6EZgMiRwfJ\n"
    "O6+c6QyhLBt2Vq2Wtl8HvgwBSiCelq6FAgMBAAGjggHkMIIB4DAfBgNVHSMEGDAW\n"
    "gBSQr2o6lFoL2JDqElZz30O0Oija5zAdBgNVHQ4EFgQU95g8ikB6kbC9vYcLhM38\n"
    "6Qm3MyUwDgYDVR0PAQH/BAQDAgWgMAwGA1UdEwEB/wQCMAAwHQYDVR0lBBYwFAYI\n"
    "KwYBBQUHAwEGCCsGAQUFBwMCMFAGA1UdIARJMEcwOwYMKwYBBAGyMQECAQMEMCsw\n"
    "KQYIKwYBBQUHAgEWHWh0dHBzOi8vc2VjdXJlLmNvbW9kby5uZXQvQ1BTMAgGBmeB\n"
    "DAECATBUBgNVHR8ETTBLMEmgR6BFhkNodHRwOi8vY3JsLmNvbW9kb2NhLmNvbS9D\n"
    "T01PRE9SU0FEb21haW5WYWxpZGF0aW9uU2VjdXJlU2VydmVyQ0EuY3JsMIGFBggr\n"
    "BgEFBQcBAQR5MHcwTwYIKwYBBQUHMAKGQ2h0dHA6Ly9jcnQuY29tb2RvY2EuY29t\n"
    "L0NPTU9ET1JTQURvbWFpblZhbGlkYXRpb25TZWN1cmVTZXJ2ZXJDQS5jcnQwJAYI\n"
    "KwYBBQUHMAGGGGh0dHA6Ly9vY3NwLmNvbW9kb2NhLmNvbTAxBgNVHREEKjAoghB2\n"
    "aWRlb21hZ2ljYWwuY29tghR3d3cudmlkZW9tYWdpY2FsLmNvbTASBgkqhkiG9w0B\n"
    "AQsfgeCB/FAAA4IBAQAaAf/vb3fp7PubQlrlDZbOYmCPmqvyf8LqEs0OxdS+GFOH\n"
    "dn+7KoggtC/SBJRoIynklTSP2Ej8pJ+1vimqmGCKGJlA7A1kV5+BfZRnsOxwqEUC\n"
    "ioN12quQp/F5ODWC6Ip/kydQd0G5ShDCddjym43LT43r8GFa4TaLifggKOS6QmK/\n"
    "VJizd8y9aQEVkQQJqgdOy24i26OXFzmO15GyOxZSesd32y1nNfNH/ofIvkMrEOoF\n"
    "kxYIZ0laVI3btEm3CaVnsUoEy5bZXjJpL58OI4/5HTJw/EyrAOao7t3bD2IUWsu+\n"
    "qIxRCJXNjDxMb93KiEfgMzGbmmPCNkzQ6AninOSC\n"
    "-----END CERTIFICATE-----\n";

static const char kMatchingSigAlgsCertString[] =
    "-----BEGIN CERTIFICATE-----\n"
    "MIIFVTCCBD2gAwIBAgIQGERW2P+5Xt15QiwP1NRmtTANBgkqhkiG9w0BAQsFADCB\n"
    "kDELMAkGA1UEBhMCR0IxGzAZBgNVBAgTEkdyZWF0ZXIgTWFuY2hlc3RlcjEQMA4G\n"
    "A1UEBxMHU2FsZm9yZDEaMBgGA1UEChMRQ09NT0RPIENBIExpbWl0ZWQxNjA0BgNV\n"
    "BAMTLUNPTU9ETyBSU0EgRG9tYWluIFZhbGlkYXRpb24gU2VjdXJlIFNlcnZlciBD\n"
    "QTAeFw0xNDA1MjEwMDAwMDBaFw0xNTA1MjkyMzU5NTlaMFQxITAfBgNVBAsTGERv\n"
    "bWFpbiBDb250cm9sIFZhbGlkYXRlZDEUMBIGA1UECxMLUG9zaXRpdmVTU0wxGTAX\n"
    "BgNVBAMTEHZpZGVvbWFnaWNhbC5jb20wggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAw\n"
    "ggEKAoIBAQDOMDIDA7wjsHNDp+cYa9On4oodRjnAxvgJQ852ahkZul+e816WzUTT\n"
    "Bs8m0LFiXjRZtsl6AHOTVbSSK3iFrke6pVdwatVP95NsR4qQaU6ITfkD9hT01vOm\n"
    "FrPv77X7RF6C0Pb8tH9ro8prpqJdTlMnjnPTQQy/ljrUaWyIQm0G1ujCApPQhQ7h\n"
    "XRZYPAk0B5jSalA1q0tjjWKohlQaQMqXpHtbofvL9hUlWw6shJdd08tUH5o0UcW3\n"
    "so0zHvVfwi4Gw6DiMc/a8aSmNJPO09Rf+xOYGB+wyMezH900OnuhMg6EZgMiRwfJ\n"
    "O6+c6QyhLBt2Vq2Wtl8HvgwBSiCelq6FAgMBAAGjggHkMIIB4DAfBgNVHSMEGDAW\n"
    "gBSQr2o6lFoL2JDqElZz30O0Oija5zAdBgNVHQ4EFgQU95g8ikB6kbC9vYcLhM38\n"
    "6Qm3MyUwDgYDVR0PAQH/BAQDAgWgMAwGA1UdEwEB/wQCMAAwHQYDVR0lBBYwFAYI\n"
    "KwYBBQUHAwEGCCsGAQUFBwMCMFAGA1UdIARJMEcwOwYMKwYBBAGyMQECAQMEMCsw\n"
    "KQYIKwYBBQUHAgEWHWh0dHBzOi8vc2VjdXJlLmNvbW9kby5uZXQvQ1BTMAgGBmeB\n"
    "DAECATBUBgNVHR8ETTBLMEmgR6BFhkNodHRwOi8vY3JsLmNvbW9kb2NhLmNvbS9D\n"
    "T01PRE9SU0FEb21haW5WYWxpZGF0aW9uU2VjdXJlU2VydmVyQ0EuY3JsMIGFBggr\n"
    "BgEFBQcBAQR5MHcwTwYIKwYBBQUHMAKGQ2h0dHA6Ly9jcnQuY29tb2RvY2EuY29t\n"
    "L0NPTU9ET1JTQURvbWFpblZhbGlkYXRpb25TZWN1cmVTZXJ2ZXJDQS5jcnQwJAYI\n"
    "KwYBBQUHMAGGGGh0dHA6Ly9vY3NwLmNvbW9kb2NhLmNvbTAxBgNVHREEKjAoghB2\n"
    "aWRlb21hZ2ljYWwuY29tghR3d3cudmlkZW9tYWdpY2FsLmNvbTANBgkqhkiG9w0B\n"
    "AQsFAAOCAQEAGgH/72936ez7m0Ja5Q2WzmJgj5qr8n/C6hLNDsXUvhhTh3Z/uyqI\n"
    "ILQv0gSUaCMp5JU0j9hI/KSftb4pqphgihiZQOwNZFefgX2UZ7DscKhFAoqDddqr\n"
    "kKfxeTg1guiKf5MnUHdBuUoQwnXY8puNy0+N6/BhWuE2i4n4ICjkukJiv1SYs3fM\n"
    "vWkBFZEECaoHTstuItujlxc5jteRsjsWUnrHd9stZzXzR/6HyL5DKxDqBZMWCGdJ\n"
    "WlSN27RJtwmlZ7FKBMuW2V4yaS+fDiOP+R0ycPxMqwDmqO7d2w9iFFrLvqiMUQiV\n"
    "zYw8TG/dyohH4DMxm5pjwjZM0OgJ4pzkgg==\n"
    "-----END CERTIFICATE-----\n";

static const char kMismatchingSigAlgsCertIssuerString[] =
    "-----BEGIN CERTIFICATE-----\n"
    "MIIGCDCCA/CgAwIBAgIQKy5u6tl1NmwUim7bo3yMBzANBgkqhkiG9w0BAQwFADCB\n"
    "hTELMAkGA1UEBhMCR0IxGzAZBgNVBAgTEkdyZWF0ZXIgTWFuY2hlc3RlcjEQMA4G\n"
    "A1UEBxMHU2FsZm9yZDEaMBgGA1UEChMRQ09NT0RPIENBIExpbWl0ZWQxKzApBgNV\n"
    "BAMTIkNPTU9ETyBSU0EgQ2VydGlmaWNhdGlvbiBBdXRob3JpdHkwHhcNMTQwMjEy\n"
    "MDAwMDAwWhcNMjkwMjExMjM1OTU5WjCBkDELMAkGA1UEBhMCR0IxGzAZBgNVBAgT\n"
    "EkdyZWF0ZXIgTWFuY2hlc3RlcjEQMA4GA1UEBxMHU2FsZm9yZDEaMBgGA1UEChMR\n"
    "Q09NT0RPIENBIExpbWl0ZWQxNjA0BgNVBAMTLUNPTU9ETyBSU0EgRG9tYWluIFZh\n"
    "bGlkYXRpb24gU2VjdXJlIFNlcnZlciBDQTCCASIwDQYJKoZIhvcNAQEBBQADggEP\n"
    "ADCCAQoCggEBAI7CAhnhoFmk6zg1jSz9AdDTScBkxwtiBUUWOqigwAwCfx3M28Sh\n"
    "bXcDow+G+eMGnD4LgYqbSRutA776S9uMIO3Vzl5ljj4Nr0zCsLdFXlIvNN5IJGS0\n"
    "Qa4Al/e+Z96e0HqnU4A7fK31llVvl0cKfIWLIpeNs4TgllfQcBhglo/uLQeTnaG6\n"
    "ytHNe+nEKpooIZFNb5JPJaXyejXdJtxGpdCsWTWM/06RQ1A/WZMebFEh7lgUq/51\n"
    "UHg+TLAchhP6a5i84DuUHoVS3AOTJBhuyydRReZw3iVDpA3hSqXttn7IzW3uLh0n\n"
    "c13cRTCAquOyQQuvvUSH2rnlG51/ruWFgqUCAwEAAaOCAWUwggFhMB8GA1UdIwQY\n"
    "MBaAFLuvfgI9+qbxPISOre44mOzZMjLUMB0GA1UdDgQWBBSQr2o6lFoL2JDqElZz\n"
    "30O0Oija5zAOBgNVHQ8BAf8EBAMCAYYwEgYDVR0TAQH/BAgwBgEB/wIBADAdBgNV\n"
    "HSUEFjAUBggrBgEFBQcDAQYIKwYBBQUHAwIwGwYDVR0gBBQwEjAGBgRVHSAAMAgG\n"
    "BmeBDAECATBMBgNVHR8ERTBDMEGgP6A9hjtodHRwOi8vY3JsLmNvbW9kb2NhLmNv\n"
    "bS9DT01PRE9SU0FDZXJ0aWZpY2F0aW9uQXV0aG9yaXR5LmNybDBxBggrBgEFBQcB\n"
    "AQRlMGMwOwYIKwYBBQUHMAKGL2h0dHA6Ly9jcnQuY29tb2RvY2EuY29tL0NPTU9E\n"
    "T1JTQUFkZFRydXN0Q0EuY3J0MCQGCCsGAQUFBzABhhhodHRwOi8vb2NzcC5jb21v\n"
    "ZG9jYS5jb20wDQYJKoZIhvcNAQEMBQADggIBAE4rdk+SHGI2ibp3wScF9BzWRJ2p\n"
    "mj6q1WZmAT7qSeaiNbz69t2Vjpk1mA42GHWx3d1Qcnyu3HeIzg/3kCDKo2cuH1Z/\n"
    "e+FE6kKVxF0NAVBGFfKBiVlsit2M8RKhjTpCipj4SzR7JzsItG8kO3KdY3RYPBps\n"
    "P0/HEZrIqPW1N+8QRcZs2eBelSaz662jue5/DJpmNXMyYE7l3YphLG5SEXdoltMY\n"
    "dVEVABt0iN3hxzgEQyjpFv3ZBdRdRydg1vs4O2xyopT4Qhrf7W8GjEXCBgCq5Ojc\n"
    "2bXhc3js9iPc0d1sjhqPpepUfJa3w/5Vjo1JXvxku88+vZbrac2/4EjxYoIQ5QxG\n"
    "V/Iz2tDIY+3GH5QFlkoakdH368+PUq4NCNk+qKBR6cGHdNXJ93SrLlP7u3r7l+L4\n"
    "HyaPs9Kg4DdbKDsx5Q5XLVq4rXmsXiBmGqW5prU5wfWYQ//u+aen/e7KJD2AFsQX\n"
    "j4rBYKEMrltDR5FL1ZoXX/nUh8HCjLfn4g8wGTeGrODcQgPmlKidrv0PJFGUzpII\n"
    "0fxQ8ANAe4hZ7Q7drNJ3gjTcBpUC2JD5Leo31Rpg0Gcg19hCC0Wvgmje3WYkN5Ap\n"
    "lBlGGSW4gNfL1IYoakRwJiNiqZ+Gb7+6kHDSVneFeO/qJakXzlByjAA6quPbYzSf\n"
    "+AZxAeKCINT+b72x\n"
    "-----END CERTIFICATE-----\n";

namespace {

class CertTest : public ::testing::Test {
 protected:
  string leaf_pem_;
  string ca_pem_;
  string ca_precert_pem_;
  string precert_pem_;
  string leaf_with_intermediate_pem_;
  string legacy_ca_pem_;

  string v2_wildcard_test5_pem_;
  string v2_wildcard_test6_pem_;
  string v2_wildcard_test7_pem_;
  string v2_wildcard_test8_pem_;
  string v2_wildcard_test9_pem_;
  string v2_wildcard_test10_pem_;
  string v2_wildcard_test11_pem_;
  string v2_wildcard_test12_pem_;
  string v2_wildcard_test13_pem_;
  string v2_wildcard_test14_pem_;
  string v2_wildcard_test15_pem_;
  string v2_wildcard_test22_pem_;
  string v2_wildcard_test23_pem_;
  string v2_wildcard_test24_pem_;

  string v2_constraint_test2_pem_;
  string v2_constraint_test3_pem_;
  string v2_constraint_test4_pem_;
  string v2_constraint_test5_pem_;
  string v2_constraint_test6_pem_;
  string v2_constraint_test7_pem_;
  string v2_constraint_test8_pem_;
  string v2_constraint_test9_pem_;

  void SetUp() {
    const string cert_dir(FLAGS_test_srcdir + "/test/testdata");
    const string cert_dir_v2(cert_dir + "/v2/");

    CHECK(util::ReadTextFile(cert_dir + "/" + kLeafCert, &leaf_pem_))
        << "Could not read test data from " << cert_dir
        << ". Wrong --test_srcdir?";
    CHECK(util::ReadTextFile(cert_dir + "/" + kCaCert, &ca_pem_));
    CHECK(util::ReadTextFile(cert_dir + "/" + kCaPreCert, &ca_precert_pem_));
    CHECK(util::ReadTextFile(cert_dir + "/" + kPreCert, &precert_pem_));
    CHECK(util::ReadTextFile(cert_dir + "/" + kLeafWithIntermediateCert,
                             &leaf_with_intermediate_pem_));
    CHECK(util::ReadTextFile(cert_dir + "/" + kLegacyCaCert, &legacy_ca_pem_));

    // V2 Wildcard redaction test certs
    CHECK(util::ReadTextFile(cert_dir_v2 + kV2WildcardRedactTest5,
                             &v2_wildcard_test5_pem_));
    CHECK(util::ReadTextFile(cert_dir_v2 + kV2WildcardRedactTest6,
                             &v2_wildcard_test6_pem_));
    CHECK(util::ReadTextFile(cert_dir_v2 + kV2WildcardRedactTest7,
                             &v2_wildcard_test7_pem_));
    CHECK(util::ReadTextFile(cert_dir_v2 + kV2WildcardRedactTest8,
                             &v2_wildcard_test8_pem_));
    CHECK(util::ReadTextFile(cert_dir_v2 + kV2WildcardRedactTest9,
                             &v2_wildcard_test9_pem_));
    CHECK(util::ReadTextFile(cert_dir_v2 + kV2WildcardRedactTest10,
                             &v2_wildcard_test10_pem_));
    CHECK(util::ReadTextFile(cert_dir_v2 + kV2WildcardRedactTest11,
                             &v2_wildcard_test11_pem_));
    CHECK(util::ReadTextFile(cert_dir_v2 + kV2WildcardRedactTest12,
                             &v2_wildcard_test12_pem_));
    CHECK(util::ReadTextFile(cert_dir_v2 + kV2WildcardRedactTest13,
                             &v2_wildcard_test13_pem_));
    CHECK(util::ReadTextFile(cert_dir_v2 + kV2WildcardRedactTest14,
                             &v2_wildcard_test14_pem_));
    CHECK(util::ReadTextFile(cert_dir_v2 + kV2WildcardRedactTest15,
                             &v2_wildcard_test15_pem_));
    CHECK(util::ReadTextFile(cert_dir_v2 + kV2WildcardRedactTest22,
                             &v2_wildcard_test22_pem_));
    CHECK(util::ReadTextFile(cert_dir_v2 + kV2WildcardRedactTest23,
                             &v2_wildcard_test23_pem_));
    CHECK(util::ReadTextFile(cert_dir_v2 + kV2WildcardRedactTest24,
                             &v2_wildcard_test24_pem_));

    // V2 Name constraint test certs
    CHECK(util::ReadTextFile(cert_dir_v2 + kV2ConstraintTest2,
                             &v2_constraint_test2_pem_));
    CHECK(util::ReadTextFile(cert_dir_v2 + kV2ConstraintTest3,
                             &v2_constraint_test3_pem_));
    CHECK(util::ReadTextFile(cert_dir_v2 + kV2ConstraintTest4,
                             &v2_constraint_test4_pem_));
    CHECK(util::ReadTextFile(cert_dir_v2 + kV2ConstraintTest5,
                             &v2_constraint_test5_pem_));
    CHECK(util::ReadTextFile(cert_dir_v2 + kV2ConstraintTest6,
                             &v2_constraint_test6_pem_));
    CHECK(util::ReadTextFile(cert_dir_v2 + kV2ConstraintTest7,
                             &v2_constraint_test7_pem_));
    CHECK(util::ReadTextFile(cert_dir_v2 + kV2ConstraintTest8,
                             &v2_constraint_test8_pem_));
    CHECK(util::ReadTextFile(cert_dir_v2 + kV2ConstraintTest9,
                             &v2_constraint_test9_pem_));
  }
};

class TbsCertificateTest : public CertTest {};
class CertChainTest : public CertTest {};

// TODO(ekasper): test encoding methods.
TEST_F(CertTest, LoadValid) {
  Cert leaf(leaf_pem_);
  EXPECT_TRUE(leaf.IsLoaded());

  Cert ca(ca_pem_);
  EXPECT_TRUE(ca.IsLoaded());

  Cert ca_pre(ca_precert_pem_);
  EXPECT_TRUE(ca_pre.IsLoaded());

  Cert pre(precert_pem_);
  EXPECT_TRUE(pre.IsLoaded());
}

TEST_F(CertTest, LoadInvalid) {
  // Bogus certs.
  Cert invalid("");
  EXPECT_FALSE(invalid.IsLoaded());
  Cert invalid2(kInvalidCertString);
  EXPECT_FALSE(invalid2.IsLoaded());
}

TEST_F(CertTest, LoadValidFromDer) {
  Cert leaf(leaf_pem_);
  string der;
  ASSERT_OK(leaf.DerEncoding(&der));
  Cert second;
  EXPECT_OK(second.LoadFromDerString(der));
  EXPECT_TRUE(second.IsLoaded());
}

TEST_F(CertTest, LoadInvalidFromDer) {
  Cert leaf(leaf_pem_);
  // Make it look almost good for extra fun.
  string der;
  ASSERT_OK(leaf.DerEncoding(&der));
  Cert second;
  EXPECT_THAT(second.LoadFromDerString(der.substr(2)),
              StatusIs(util::error::INVALID_ARGUMENT));
  EXPECT_FALSE(second.IsLoaded());
}

TEST_F(CertTest, PrintSubjectName) {
  Cert leaf(leaf_pem_);
  EXPECT_EQ("C=GB, O=Certificate Transparency, ST=Wales, L=Erw Wen",
            leaf.PrintSubjectName());
}

TEST_F(CertTest, PrintIssuerName) {
  Cert leaf(leaf_pem_);
  EXPECT_EQ("C=GB, O=Certificate Transparency CA, ST=Wales, L=Erw Wen",
            leaf.PrintIssuerName());
}

TEST_F(CertTest, PrintNotBefore) {
  Cert leaf(leaf_pem_);
  EXPECT_EQ("Jun  1 00:00:00 2012 GMT", leaf.PrintNotBefore());
}

TEST_F(CertTest, PrintNotAfter) {
  Cert leaf(leaf_pem_);
  EXPECT_EQ("Jun  1 00:00:00 2022 GMT", leaf.PrintNotAfter());
}

TEST_F(CertTest, PrintSignatureAlgorithm) {
  Cert leaf(leaf_pem_);
  EXPECT_EQ("sha1WithRSAEncryption", leaf.PrintSignatureAlgorithm());
}

TEST_F(CertTest, TestUnsupportedAlgorithm) {
  Cert legacy(legacy_ca_pem_);
  ASSERT_EQ("md2WithRSAEncryption", legacy.PrintSignatureAlgorithm());
// MD2 is disabled by default on modern OpenSSL and you should be surprised to
// see anything else. Make the test fail if this is not the case to notify the
// user that their setup is insecure.
#ifdef OPENSSL_NO_MD2
  EXPECT_THAT(legacy.IsSignedBy(legacy).status(),
              StatusIs(Code::UNIMPLEMENTED));
#else
  LOG(WARNING) << "Skipping test: MD2 is enabled! You should configure "
               << "OpenSSL with -DOPENSSL_NO_MD2 to be safe!";
#endif
}

TEST_F(CertTest, Identical) {
  Cert leaf(leaf_pem_);
  Cert ca(ca_pem_);
  EXPECT_TRUE(leaf.IsIdenticalTo(leaf));
  EXPECT_FALSE(leaf.IsIdenticalTo(ca));
  EXPECT_FALSE(ca.IsIdenticalTo(leaf));
}

TEST_F(CertTest, Extensions) {
  Cert leaf(leaf_pem_);
  Cert ca(ca_pem_);
  Cert ca_pre(ca_precert_pem_);
  Cert pre(precert_pem_);


  // Some facts we know are true about those test certs.
  EXPECT_TRUE(leaf.HasExtension(NID_authority_key_identifier).ValueOrDie());
  EXPECT_FALSE(
      leaf.HasCriticalExtension(NID_authority_key_identifier).ValueOrDie());
  EXPECT_TRUE(pre.HasCriticalExtension(cert_trans::NID_ctPoison).ValueOrDie());

  EXPECT_FALSE(leaf.HasBasicConstraintCATrue().ValueOrDie());
  EXPECT_TRUE(ca.HasBasicConstraintCATrue().ValueOrDie());
  EXPECT_TRUE(
      ca_pre.HasExtendedKeyUsage(cert_trans::NID_ctPrecertificateSigning)
          .ValueOrDie());
}

TEST_F(CertTest, Issuers) {
  Cert leaf(leaf_pem_);
  Cert ca(ca_pem_);
  Cert ca_pre(ca_precert_pem_);
  Cert pre(precert_pem_);

  EXPECT_TRUE(leaf.IsIssuedBy(ca).ValueOrDie());
  EXPECT_TRUE(leaf.IsSignedBy(ca).ValueOrDie());

  EXPECT_FALSE(ca.IsIssuedBy(leaf).ValueOrDie());
  EXPECT_FALSE(ca.IsSignedBy(leaf).ValueOrDie());

  EXPECT_FALSE(leaf.IsSelfSigned().ValueOrDie());
  EXPECT_TRUE(ca.IsSelfSigned().ValueOrDie());
}

TEST_F(CertTest, DerEncodedNames) {
  Cert leaf(leaf_pem_);
  Cert ca(ca_pem_);

  ASSERT_TRUE(leaf.IsIssuedBy(ca).ValueOrDie());

  string leaf_subject, leaf_issuer, ca_subject, ca_issuer;
  EXPECT_OK(leaf.DerEncodedSubjectName(&leaf_subject));
  EXPECT_FALSE(leaf_subject.empty());

  EXPECT_OK(leaf.DerEncodedIssuerName(&leaf_issuer));
  EXPECT_FALSE(leaf_issuer.empty());

  EXPECT_OK(ca.DerEncodedSubjectName(&ca_subject));
  EXPECT_FALSE(ca_subject.empty());

  EXPECT_OK(ca.DerEncodedIssuerName(&ca_issuer));
  EXPECT_FALSE(ca_issuer.empty());

  EXPECT_EQ(leaf_issuer, ca_subject);
  EXPECT_EQ(ca_subject, ca_issuer);
  EXPECT_NE(leaf_subject, leaf_issuer);
}

TEST_F(CertTest, SignatureAlgorithmMatches) {
  Cert matching_algs(kMatchingSigAlgsCertString);
  Cert issuer(kMismatchingSigAlgsCertIssuerString);
  EXPECT_TRUE(matching_algs.IsSignedBy(issuer).ValueOrDie());
  Cert mismatched_algs(kMismatchingSigAlgsCertString);
  EXPECT_FALSE(mismatched_algs.IsSignedBy(issuer).ValueOrDie());
}

TEST_F(CertTest, TestIsRedactedHost) {
  EXPECT_FALSE(cert_trans::IsRedactedHost(""));
  EXPECT_FALSE(cert_trans::IsRedactedHost("example.com"));

  EXPECT_TRUE(cert_trans::IsRedactedHost("?.example.com"));
  EXPECT_TRUE(cert_trans::IsRedactedHost("?.?.example.com"));
  EXPECT_TRUE(cert_trans::IsRedactedHost("top.?.example.com"));
}

TEST_F(CertTest, TestIsValidRedactedHost) {
  EXPECT_TRUE(cert_trans::IsValidRedactedHost("?.example.com"));
  EXPECT_TRUE(cert_trans::IsValidRedactedHost("?.?.example.com"));
  EXPECT_TRUE(cert_trans::IsValidRedactedHost("*.?.example.com"));
  EXPECT_TRUE(cert_trans::IsValidRedactedHost("*.?.?.example.com"));

  EXPECT_FALSE(cert_trans::IsValidRedactedHost("top.?.example.com"));
  EXPECT_FALSE(cert_trans::IsValidRedactedHost("top.secret.example.?"));
  EXPECT_FALSE(cert_trans::IsValidRedactedHost("top.secret.?.com"));
  EXPECT_FALSE(cert_trans::IsValidRedactedHost("top.*.secret.?.com"));
  EXPECT_FALSE(cert_trans::IsValidRedactedHost("?.*.example.com"));
  EXPECT_FALSE(cert_trans::IsValidRedactedHost("*.secret.?.com"));
}

TEST_F(CertTest, TestNoWildcardRedactionIsValid) {
  Cert leaf(leaf_pem_);
  EXPECT_OK(leaf.IsValidWildcardRedaction());
}

TEST_F(CertTest, TestWildcardRedactTestCase5) {
  // This is invalid because the CN is redacted, no DNS or extension
  Cert leaf(v2_wildcard_test5_pem_);
  EXPECT_THAT(leaf.IsValidWildcardRedaction(),
              StatusIs(util::error::INVALID_ARGUMENT));
}

TEST_F(CertTest, TestWildcardRedactTestCase6) {
  // This is invalid because the CN differs from the first DNS-ID
  Cert leaf(v2_wildcard_test6_pem_);
  EXPECT_THAT(leaf.IsValidWildcardRedaction(),
              StatusIs(util::error::INVALID_ARGUMENT));
}

TEST_F(CertTest, TestWildcardRedactTestCase7) {
  // This should be a valid redaction of 1 label with everything set
  // correctly in the extension
  Cert leaf(v2_wildcard_test7_pem_);
  EXPECT_OK(leaf.IsValidWildcardRedaction());
}

TEST_F(CertTest, TestWildcardRedactTestCase8) {
  // This should be a valid redaction of 1 label with everything set
  // correctly in the extension and a '*' at left of name.
  Cert leaf(v2_wildcard_test8_pem_);
  EXPECT_OK(leaf.IsValidWildcardRedaction());
}

TEST_F(CertTest, TestWildcardRedactTestCase9) {
  // Should be invalid as the redacted label does not follow RFC rules
  Cert leaf(v2_wildcard_test9_pem_);
  EXPECT_THAT(leaf.IsValidWildcardRedaction(),
              StatusIs(util::error::INVALID_ARGUMENT));
}

TEST_F(CertTest, TestWildcardRedactTestCase10) {
  // Should be invalid as redacted label uses '*' incorrectly
  Cert leaf(v2_wildcard_test10_pem_);
  EXPECT_THAT(leaf.IsValidWildcardRedaction(),
              StatusIs(util::error::INVALID_ARGUMENT));
}

TEST_F(CertTest, TestWildcardRedactTestCase11) {
  // Should be invalid as there are too many label count values
  Cert leaf(v2_wildcard_test11_pem_);
  EXPECT_THAT(leaf.IsValidWildcardRedaction(),
              StatusIs(util::error::INVALID_ARGUMENT));
}

TEST_F(CertTest, TestWildcardRedactTestCase12) {
  // This should be invalid because the CT extension contains -ve value
  Cert leaf(v2_wildcard_test12_pem_);
  EXPECT_THAT(leaf.IsValidWildcardRedaction(),
              StatusIs(util::error::INVALID_ARGUMENT));
}

TEST_F(CertTest, TestWildcardRedactTestCase13) {
  Cert leaf(v2_wildcard_test13_pem_);
  EXPECT_OK(leaf.IsValidWildcardRedaction());
}

TEST_F(CertTest, TestWildcardRedactTestCase14) {
  Cert leaf(v2_wildcard_test14_pem_);
  EXPECT_OK(leaf.IsValidWildcardRedaction());
}

TEST_F(CertTest, TestWildcardRedactTestCase15) {
  // This should be invalid because the CT extension has too many values
  Cert leaf(v2_wildcard_test15_pem_);
  EXPECT_THAT(leaf.IsValidWildcardRedaction(),
              StatusIs(util::error::INVALID_ARGUMENT));
}

TEST_F(CertTest, TestWildcardRedactTestCase22) {
  // This should be a redaction of 1 label but no extension required by
  // RFC section 3.2.2
  Cert leaf(v2_wildcard_test22_pem_);
  EXPECT_THAT(leaf.IsValidWildcardRedaction(),
              StatusIs(util::error::INVALID_ARGUMENT));
}

TEST_F(CertTest, TestWildcardRedactTestCase23) {
  // Should not be valid because the CT extension is not a SEQUENCE OF
  // type
  Cert leaf(v2_wildcard_test23_pem_);
  EXPECT_THAT(leaf.IsValidWildcardRedaction(),
              StatusIs(util::error::INVALID_ARGUMENT));
}

TEST_F(CertTest, TestWildcardRedactTestCase24) {
  // Should not be valid because not all the items in the CT extension sequence
  // are ASN1_INTEGER type
  Cert leaf(v2_wildcard_test24_pem_);
  EXPECT_THAT(leaf.IsValidWildcardRedaction(),
              StatusIs(util::error::INVALID_ARGUMENT));
}

TEST_F(CertTest, TestConstraintTestCase2) {
  // This should be valid as the cert is non CA and the checks do not apply
  Cert leaf(v2_constraint_test2_pem_);
  EXPECT_OK(leaf.IsValidNameConstrainedIntermediateCa());
}

TEST_F(CertTest, TestConstraintTestCase3) {
  // This should be valid as the cert is CA but has no name constraint
  Cert leaf(v2_constraint_test3_pem_);
  EXPECT_OK(leaf.IsValidNameConstrainedIntermediateCa());
}

TEST_F(CertTest, TestConstraintTestCase4) {
  // Not valid as there is a constraint but no CT ext
  Cert leaf(v2_constraint_test4_pem_);
  EXPECT_THAT(leaf.IsValidNameConstrainedIntermediateCa(),
              StatusIs(util::error::INVALID_ARGUMENT));
}

TEST_F(CertTest, TestConstraintTestCase5) {
  // Not valid as there is no DNS entry in name constraints
  Cert leaf(v2_constraint_test5_pem_);
  EXPECT_THAT(leaf.IsValidNameConstrainedIntermediateCa(),
              StatusIs(util::error::INVALID_ARGUMENT));
}

TEST_F(CertTest, TestConstraintTestCase6) {
  // This should be valid as the CA cert contains valid name constraints +
  // CT extension
  Cert leaf(v2_constraint_test6_pem_);
  EXPECT_OK(leaf.IsValidNameConstrainedIntermediateCa());
}

TEST_F(CertTest, TestConstraintTestCase7) {
  // This should be valid as the CA cert contains valid name constraints +
  // CT extension + multiple DNS entries
  Cert leaf(v2_constraint_test7_pem_);
  EXPECT_OK(leaf.IsValidNameConstrainedIntermediateCa());
}

TEST_F(CertTest, TestConstraintTestCase8) {
  // This should be invalid as there is no IP exclusion in name constraint
  Cert leaf(v2_constraint_test8_pem_);
  EXPECT_THAT(leaf.IsValidNameConstrainedIntermediateCa(),
              StatusIs(util::error::INVALID_ARGUMENT));
}

TEST_F(CertTest, TestConstraintTestCase9) {
  // This should be invalid as both IPv4 and v6 ranges not excluded
  Cert leaf(v2_constraint_test9_pem_);
  EXPECT_THAT(leaf.IsValidNameConstrainedIntermediateCa(),
              StatusIs(util::error::INVALID_ARGUMENT));
}

TEST_F(TbsCertificateTest, DerEncoding) {
  Cert leaf(leaf_pem_);
  TbsCertificate tbs(leaf);

  string cert_tbs_der, raw_tbs_der;
  EXPECT_OK(leaf.DerEncodedTbsCertificate(&cert_tbs_der));
  EXPECT_OK(tbs.DerEncoding(&raw_tbs_der));
  EXPECT_EQ(cert_tbs_der, raw_tbs_der);
}

TEST_F(TbsCertificateTest, DeleteExtension) {
  Cert leaf(leaf_pem_);

  ASSERT_TRUE(leaf.HasExtension(NID_authority_key_identifier).ValueOrDie());

  TbsCertificate tbs(leaf);
  string der_before, der_after;
  EXPECT_OK(tbs.DerEncoding(&der_before));
  EXPECT_OK(tbs.DeleteExtension(NID_authority_key_identifier));
  EXPECT_OK(tbs.DerEncoding(&der_after));
  EXPECT_NE(der_before, der_after);

  ASSERT_FALSE(leaf.HasExtension(cert_trans::NID_ctPoison).ValueOrDie());
  TbsCertificate tbs2(leaf);
  string der_before2, der_after2;
  EXPECT_OK(tbs2.DerEncoding(&der_before2));
  EXPECT_THAT(tbs2.DeleteExtension(cert_trans::NID_ctPoison),
              StatusIs(util::error::NOT_FOUND));
  EXPECT_OK(tbs2.DerEncoding(&der_after2));
  EXPECT_EQ(der_before2, der_after2);
}

TEST_F(TbsCertificateTest, CopyIssuer) {
  Cert leaf(leaf_pem_);
  Cert different(leaf_with_intermediate_pem_);

  TbsCertificate tbs(leaf);
  string der_before, der_after;
  EXPECT_OK(tbs.DerEncoding(&der_before));
  EXPECT_OK(tbs.CopyIssuerFrom(different));
  EXPECT_OK(tbs.DerEncoding(&der_after));
  EXPECT_NE(der_before, der_after);

  TbsCertificate tbs2(leaf);
  string der_before2, der_after2;
  EXPECT_OK(tbs2.DerEncoding(&der_before2));
  EXPECT_OK(tbs2.CopyIssuerFrom(leaf));
  EXPECT_OK(tbs2.DerEncoding(&der_after2));
  EXPECT_EQ(der_before2, der_after2);
}

TEST_F(CertChainTest, LoadValid) {
  // A single certificate.
  CertChain chain(leaf_pem_);
  EXPECT_TRUE(chain.IsLoaded());
  EXPECT_EQ(chain.Length(), 1U);

  CertChain chain2(leaf_pem_ + ca_pem_);
  EXPECT_TRUE(chain2.IsLoaded());
  EXPECT_EQ(chain2.Length(), 2U);
}

TEST_F(CertChainTest, LoadInvalid) {
  // A single certificate.
  CertChain chain("bogus");
  EXPECT_FALSE(chain.IsLoaded());
  EXPECT_EQ(chain.Length(), 0U);

  CertChain chain2(leaf_pem_ + string(kInvalidCertString));
  EXPECT_FALSE(chain.IsLoaded());
  EXPECT_EQ(chain.Length(), 0U);
}

TEST_F(CertChainTest, AddCert) {
  CertChain chain(leaf_pem_);
  EXPECT_EQ(chain.Length(), 1U);

  chain.AddCert(new Cert(ca_pem_));
  EXPECT_EQ(chain.Length(), 2U);

  chain.AddCert(NULL);
  EXPECT_EQ(chain.Length(), 2U);

  chain.AddCert(new Cert("bogus"));
  EXPECT_EQ(chain.Length(), 2U);
}

TEST_F(CertChainTest, RemoveCert) {
  CertChain chain(leaf_pem_);
  EXPECT_EQ(chain.Length(), 1U);
  chain.RemoveCert();
  EXPECT_EQ(0U, chain.Length());

  // Does nothing.
  chain.RemoveCert();
  EXPECT_EQ(0U, chain.Length());
}

TEST_F(CertChainTest, IssuerChains) {
  // A single certificate.
  CertChain chain(leaf_pem_);
  EXPECT_OK(chain.IsValidCaIssuerChainMaybeLegacyRoot());
  EXPECT_OK(chain.IsValidSignatureChain());

  // Two certs.
  CertChain chain2(leaf_pem_ + ca_pem_);
  EXPECT_OK(chain.IsValidCaIssuerChainMaybeLegacyRoot());
  EXPECT_OK(chain.IsValidSignatureChain());

  // In reverse order.
  CertChain chain3(ca_pem_ + leaf_pem_);
  EXPECT_THAT(chain3.IsValidCaIssuerChainMaybeLegacyRoot(),
              StatusIs(util::error::INVALID_ARGUMENT));
  EXPECT_THAT(chain3.IsValidSignatureChain(),
              StatusIs(util::error::INVALID_ARGUMENT));

  // Invalid
  CertChain invalid("");
  EXPECT_THAT(invalid.IsValidCaIssuerChainMaybeLegacyRoot(),
              StatusIs(util::error::FAILED_PRECONDITION));
  EXPECT_THAT(invalid.IsValidSignatureChain(),
              StatusIs(util::error::FAILED_PRECONDITION));
}

TEST_F(CertChainTest, PreCertChain) {
  // A precert chain.
  string pem_bundle = precert_pem_ + ca_pem_;
  PreCertChain pre_chain(pem_bundle);
  ASSERT_TRUE(pre_chain.IsLoaded());
  EXPECT_EQ(pre_chain.Length(), 2U);
  EXPECT_OK(pre_chain.IsValidCaIssuerChainMaybeLegacyRoot());
  EXPECT_OK(pre_chain.IsValidSignatureChain());
  EXPECT_TRUE(pre_chain.IsWellFormed().ValueOrDie());

  // Try to construct a precert chain from regular certs.
  // The chain should load, but is not well-formed.
  pem_bundle = leaf_pem_ + ca_pem_;
  PreCertChain pre_chain2(pem_bundle);
  ASSERT_TRUE(pre_chain2.IsLoaded());
  EXPECT_EQ(pre_chain2.Length(), 2U);
  EXPECT_OK(pre_chain2.IsValidCaIssuerChainMaybeLegacyRoot());
  EXPECT_OK(pre_chain2.IsValidSignatureChain());
  EXPECT_FALSE(pre_chain2.IsWellFormed().ValueOrDie());
}

}  // namespace

int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  OpenSSL_add_all_algorithms();
  ERR_load_crypto_strings();
  cert_trans::LoadCtExtensions();
  return RUN_ALL_TESTS();
}
