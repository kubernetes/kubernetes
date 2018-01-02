package bundler

// This test file contains tests on checking the correctness of BundleFromFile and Bundle.
// We simulate various scenarios for Bundle and funnel the tests through BundleFromFile.
import (
	"encoding/json"
	"testing"
)

// A helper structure that defines a BundleFromFile test case.
type fileTest struct {
	// PEM cert file to be bundled
	cert string
	// PEM private key file to be bundled
	key string
	// Root CA bundle
	caBundleFile string
	// Trust intermediate bundle
	intBundleFile string
	// Additional PEM intermediate certificates to be added into the bundler
	extraIntermediates string
	// Bundler creation function
	bundlerConstructor func(*testing.T) (b *Bundler)
	// Error checking function
	errorCallback func(*testing.T, error)
	// Bundle checking function
	bundleChecking func(*testing.T, *Bundle)
}

/* ========== BundleFromFile Test Setup =============

For each pair of crypto algorithm X and key size Y, a CA chain is constructed:
	Test_root_CA -> inter-L1 -> inter-L2--> cfssl-leaf-ecdsa256
	                                    |-> cfssl-leaf-ecdsa384
	                                    |-> cfssl-leaf-ecdsa521
	                                    |-> cfssl-leaf-rsa2048
	                                    |-> cfssl-leaf-rsa3072
	                                    |-> cfssl-leaf-rsa4096

Test_root_CA is a RSA cert, inter-L1 is RSA 4096 cert, inter-L2 is ecdsa-384 cert.

The max path length is set to be 1 for non-root CAs.
Two inter-* certs are assembled in intermediates.crt

There is also an expired L1 cert, sharing the same CSR with inter-L1. Also the
root CA processes the inter-L2 CSR directly to generate inter-L2-direct cert.
*	Test_root_CA--> inter-L1-expired
	            |-> inter-L2-direct
Using inter-L2-direct as additional intermediate cert should shorten the
bundle chain.
*/
const (
	leafECDSA256    = "testdata/cfssl-leaf-ecdsa256.pem"
	leafECDSA384    = "testdata/cfssl-leaf-ecdsa384.pem"
	leafECDSA521    = "testdata/cfssl-leaf-ecdsa521.pem"
	leafRSA2048     = "testdata/cfssl-leaf-rsa2048.pem"
	leafRSA3072     = "testdata/cfssl-leaf-rsa3072.pem"
	leafRSA4096     = "testdata/cfssl-leaf-rsa4096.pem"
	leafKeyECDSA256 = "testdata/cfssl-leaf-ecdsa256.key"
	leafKeyECDSA384 = "testdata/cfssl-leaf-ecdsa384.key"
	leafKeyECDSA521 = "testdata/cfssl-leaf-ecdsa521.key"
	leafKeyRSA2048  = "testdata/cfssl-leaf-rsa2048.key"
	leafKeyRSA3072  = "testdata/cfssl-leaf-rsa3072.key"
	leafKeyRSA4096  = "testdata/cfssl-leaf-rsa4096.key"
	leafletRSA4096  = "testdata/cfssl-leaflet-rsa4096.pem"
	interL1         = "testdata/inter-L1.pem"
	interL1Expired  = "testdata/inter-L1-expired.pem"
	interL1CSR      = "testdata/inter-L1.csr"
	interL2         = "testdata/inter-L2.pem"

	interL2Direct = "testdata/inter-L2-direct.pem"
	partialBundle = "testdata/partial-bundle.pem"         // partialBundle is a partial cert chain {leaf-ecds256,  inter-L2}
	rpBundle      = "testdata/reverse-partial-bundle.pem" // partialBundle is a partial cert chain in the reverse order {inter-L2, leaf-ecdsa256}
	badBundle     = "testdata/bad-bundle.pem"             // badBundle is a non-verifying partial bundle {leaf-ecdsa256, leaf-ecdsa384}
	interL2CSR    = "testdata/inter-L2.csr"
	certDSA2048   = "testdata/dsa2048.pem"
	keyDSA2048    = "testdata/dsa2048.key"
)

// BundleFromFile test cases.
var fileTests = []fileTest{
	// Input verification
	{
		cert:          "not_such_cert.pem",
		caBundleFile:  testCFSSLRootBundle,
		intBundleFile: testCFSSLIntBundle,
		errorCallback: ExpectErrorMessage(`"code":1001`),
	},
	{
		cert:          emptyPEM,
		caBundleFile:  testCFSSLRootBundle,
		intBundleFile: testCFSSLIntBundle,
		errorCallback: ExpectErrorMessage(`"code":1002`),
	},

	// Normal Keyless bundling for all supported public key types
	{
		cert:           leafECDSA256,
		caBundleFile:   testCFSSLRootBundle,
		intBundleFile:  testCFSSLIntBundle,
		errorCallback:  nil,
		bundleChecking: ExpectBundleLength(3),
	},
	{
		cert:           leafECDSA384,
		caBundleFile:   testCFSSLRootBundle,
		intBundleFile:  testCFSSLIntBundle,
		errorCallback:  nil,
		bundleChecking: ExpectBundleLength(3),
	},
	{
		cert:           leafECDSA521,
		caBundleFile:   testCFSSLRootBundle,
		intBundleFile:  testCFSSLIntBundle,
		errorCallback:  nil,
		bundleChecking: ExpectBundleLength(3),
	},
	{
		cert:           leafRSA2048,
		caBundleFile:   testCFSSLRootBundle,
		intBundleFile:  testCFSSLIntBundle,
		errorCallback:  nil,
		bundleChecking: ExpectBundleLength(3),
	},
	{
		cert:           leafRSA3072,
		caBundleFile:   testCFSSLRootBundle,
		intBundleFile:  testCFSSLIntBundle,
		errorCallback:  nil,
		bundleChecking: ExpectBundleLength(3),
	},
	{
		cert:           leafRSA4096,
		caBundleFile:   testCFSSLRootBundle,
		intBundleFile:  testCFSSLIntBundle,
		errorCallback:  nil,
		bundleChecking: ExpectBundleLength(3),
	},

	// Normal bundling with private key for all supported key types
	{
		cert:           leafECDSA256,
		key:            leafKeyECDSA256,
		caBundleFile:   testCFSSLRootBundle,
		intBundleFile:  testCFSSLIntBundle,
		errorCallback:  nil,
		bundleChecking: ExpectBundleLength(3),
	},
	{
		cert:           leafECDSA384,
		key:            leafKeyECDSA384,
		caBundleFile:   testCFSSLRootBundle,
		intBundleFile:  testCFSSLIntBundle,
		errorCallback:  nil,
		bundleChecking: ExpectBundleLength(3),
	},
	{
		cert:           leafECDSA521,
		key:            leafKeyECDSA521,
		caBundleFile:   testCFSSLRootBundle,
		intBundleFile:  testCFSSLIntBundle,
		errorCallback:  nil,
		bundleChecking: ExpectBundleLength(3),
	},
	{
		cert:           leafRSA2048,
		key:            leafKeyRSA2048,
		caBundleFile:   testCFSSLRootBundle,
		intBundleFile:  testCFSSLIntBundle,
		errorCallback:  nil,
		bundleChecking: ExpectBundleLength(3),
	},
	{
		cert:           leafRSA3072,
		key:            leafKeyRSA3072,
		caBundleFile:   testCFSSLRootBundle,
		intBundleFile:  testCFSSLIntBundle,
		errorCallback:  nil,
		bundleChecking: ExpectBundleLength(3),
	},
	{
		cert:           leafRSA4096,
		key:            leafKeyRSA4096,
		caBundleFile:   testCFSSLRootBundle,
		intBundleFile:  testCFSSLIntBundle,
		errorCallback:  nil,
		bundleChecking: ExpectBundleLength(3),
	},

	// Bundling with errors

	// leaflet cert is signed by a leaf cert which is not included the intermediate bundle.
	// So an UnknownAuthority error is expected.
	{
		cert:          leafletRSA4096,
		caBundleFile:  testCFSSLRootBundle,
		intBundleFile: testCFSSLIntBundle,
		errorCallback: ExpectErrorMessage(`"code":1220`),
	},
	// Expect TooManyIntermediates error because max path length is 1 for
	// inter-L1 but the leaflet cert is 2 CA away from inter-L1.
	{
		cert:               leafletRSA4096,
		extraIntermediates: leafRSA4096,
		caBundleFile:       testCFSSLRootBundle,
		intBundleFile:      testCFSSLIntBundle,
		errorCallback:      ExpectErrorMessage(`"code":1213`),
	},
	// Bundle with expired inter-L1 intermediate cert only, expect error 1211 VerifyFailed:Expired.
	{
		cert:               interL2,
		extraIntermediates: interL1Expired,
		caBundleFile:       testCFSSLRootBundle,
		intBundleFile:      emptyPEM,
		errorCallback:      ExpectErrorMessage(`"code":1211`),
	},

	// Bundle with private key mismatch
	// RSA cert, ECC private key
	{
		cert:          leafRSA4096,
		key:           leafKeyECDSA256,
		caBundleFile:  testCFSSLRootBundle,
		intBundleFile: testCFSSLIntBundle,
		errorCallback: ExpectErrorMessages([]string{`"code":2300,`, `"message":"Private key does not match public key"`}),
	},
	// ECC cert, RSA private key
	{
		cert:          leafECDSA256,
		key:           leafKeyRSA4096,
		caBundleFile:  testCFSSLRootBundle,
		intBundleFile: testCFSSLIntBundle,
		errorCallback: ExpectErrorMessages([]string{`"code":2300,`, `"message":"Private key does not match public key"`}),
	},
	// RSA 2048 cert, RSA 4096  private key
	{
		cert:          leafRSA2048,
		key:           leafKeyRSA4096,
		caBundleFile:  testCFSSLRootBundle,
		intBundleFile: testCFSSLIntBundle,
		errorCallback: ExpectErrorMessages([]string{`"code":2300,`, `"message":"Private key does not match public key"`}),
	},
	// ECDSA 256 cert, ECDSA 384  private key
	{
		cert:          leafECDSA256,
		key:           leafKeyECDSA384,
		caBundleFile:  testCFSSLRootBundle,
		intBundleFile: testCFSSLIntBundle,
		errorCallback: ExpectErrorMessages([]string{`"code":2300,`, `"message":"Private key does not match public key"`}),
	},

	// DSA is NOT supported.
	// Keyless bundling, expect private key error "NotRSAOrECC"
	{
		cert:          certDSA2048,
		caBundleFile:  testCFSSLRootBundle,
		intBundleFile: testCFSSLIntBundle,
		errorCallback: ExpectErrorMessages([]string{`"code":2200,`, `"message":"Private key algorithm is not RSA or ECC"`}),
	},
	// Bundling with DSA private key, expect error "Failed to parse private key"
	{
		cert:          certDSA2048,
		key:           keyDSA2048,
		caBundleFile:  testCFSSLRootBundle,
		intBundleFile: testCFSSLIntBundle,
		errorCallback: ExpectErrorMessages([]string{`"code":2003,`, `"message":"Failed to parse private key"`}),
	},

	// Bundle with partial chain less some intermediates, expected error 1220: UnknownAuthority
	{
		cert:          badBundle,
		caBundleFile:  testCFSSLRootBundle,
		intBundleFile: interL1,
		errorCallback: ExpectErrorMessage(`"code":1220`),
	},

	// Bundle with misplaced key as cert
	{
		cert:          leafKeyECDSA256,
		caBundleFile:  testCFSSLRootBundle,
		intBundleFile: testCFSSLIntBundle,
		errorCallback: ExpectErrorMessages([]string{`"code":1003,`, `"message":"Failed to parse certificate"`}),
	},

	// Bundle with misplaced cert as key
	{
		cert:          leafECDSA256,
		key:           leafECDSA256,
		caBundleFile:  testCFSSLRootBundle,
		intBundleFile: testCFSSLIntBundle,
		errorCallback: ExpectErrorMessages([]string{`"code":2003,`, `"message":"Failed to parse private key"`}),
	},

	// Smart Bundling
	// Bundling with a partial bundle should work the same as bundling the leaf.
	{
		cert:           partialBundle,
		caBundleFile:   testCFSSLRootBundle,
		intBundleFile:  testCFSSLIntBundle,
		errorCallback:  nil,
		bundleChecking: ExpectBundleLength(3),
	},

	// Bundle with a partial bundle such that the intermediate provided in the
	// partial bundle is verify by an intermediate. Yet itself is not in the intermediate
	// pool. In such cases, the bundling should be able to store the new intermediate
	// and return a correct bundle.
	{
		cert:           partialBundle,
		caBundleFile:   testCFSSLRootBundle,
		intBundleFile:  interL1,
		errorCallback:  nil,
		bundleChecking: ExpectBundleLength(3),
	},

	// Bundle with a reverse-ordered partial bundle.
	// Bundler should be able to detect it and return a correct bundle.
	{
		cert:           rpBundle,
		caBundleFile:   testCFSSLRootBundle,
		intBundleFile:  interL1,
		errorCallback:  nil,
		bundleChecking: ExpectBundleLength(3),
	},

	// Bundle with a L2 cert direct signed by root, expect a shorter chain of length 2.
	{
		cert:               leafECDSA256,
		extraIntermediates: interL2Direct,
		caBundleFile:       testCFSSLRootBundle,
		intBundleFile:      testCFSSLIntBundle,
		errorCallback:      nil,
		bundleChecking:     ExpectBundleLength(2),
	},
}

// TestBundleFromFile goes through test cases defined in fileTests. See below for test cases definition and details.
func TestBundleFromFile(t *testing.T) {
	for _, test := range fileTests {
		b := newCustomizedBundlerFromFile(t, test.caBundleFile, test.intBundleFile, test.extraIntermediates)
		bundle, err := b.BundleFromFile(test.cert, test.key, Optimal, "")
		if test.errorCallback != nil {
			test.errorCallback(t, err)
		} else {
			if err != nil {
				t.Fatalf("expected no error. but an error occurred: %v", err)
			}
			if test.bundleChecking != nil {
				test.bundleChecking(t, bundle)
			}
		}

		if bundle != nil {
			bundle.Cert = nil
			if _, err = json.Marshal(bundle); err == nil {
				t.Fatal("bundle should fail with no cert")
			}
		}
	}
}
