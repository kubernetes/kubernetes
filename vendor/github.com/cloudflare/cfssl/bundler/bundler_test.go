package bundler

// This test file contains mostly tests on checking Bundle.Status when bundling under different circumstances.
import (
	"bytes"
	"crypto/x509"
	"encoding/json"
	"io/ioutil"
	"strings"
	"testing"
	"time"

	"github.com/cloudflare/cfssl/config"
	"github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/signer"
	"github.com/cloudflare/cfssl/signer/local"
	"github.com/cloudflare/cfssl/ubiquity"
)

const (
	testCaBundle        = "testdata/ca-bundle.pem"
	testIntCaBundle     = "testdata/int-bundle.pem"
	testNSSRootBundle   = "testdata/nss.pem"
	testMetadata        = "testdata/ca-bundle.crt.metadata"
	testCFSSLRootBundle = "testdata/ca.pem"
	testCAFile          = "testdata/ca.pem"
	testCAKeyFile       = "testdata/ca.key"
	testCFSSLIntBundle  = "testdata/intermediates.crt"
	emptyPEM            = "testdata/empty.pem"
	interL1SHA1         = "testdata/inter-L1-sha1.pem"
	interL1Key          = "testdata/inter-L1.key"
	interL2SHA2         = "testdata/inter-L2.pem"
	interL2Key          = "testdata/inter-L2.key"
)

// Simply create a bundler
func TestNewBundler(t *testing.T) {
	newBundler(t)
}

func TestNewBundlerMissingCA(t *testing.T) {
	badFile := "testdata/no_such_file.pem"
	_, err := NewBundler(badFile, testIntCaBundle)
	if err == nil {
		t.Fatal("Should fail with error code 4001")
	}

	// generate a function checking error content
	errorCheck := ExpectErrorMessage(`"code":4001`)
	errorCheck(t, err)
}

func TestNewBundlerMissingIntermediate(t *testing.T) {
	badFile := "testdata/no_such_file.pem"
	_, err := NewBundler(testCaBundle, badFile)
	if err == nil {
		t.Fatal("Should fail with error code 3001")
	}

	// generate a function checking error content
	errorCheck := ExpectErrorMessage(`"code":3001`)
	errorCheck(t, err)
}

// JSON object of a bundle
type bundleObject struct {
	Bundle      string   `json:"bundle"`
	Root        string   `json:"root"`
	Cert        string   `json:"crt"`
	Key         string   `json:"key"`
	KeyType     string   `json:"key_type"`
	KeySize     int      `json:"key_size"`
	Issuer      string   `json:"issuer"`
	Subject     string   `json:"subject"`
	Expires     string   `json:"expires"`
	Hostnames   []string `json:"hostnames"`
	OCSPSupport bool     `json:"ocsp_support"`
	CRLSupport  bool     `json:"crl_support"`
	OCSP        []string `json:"ocsp"`
	Signature   string   `json:"signature"`
	Status      BundleStatus
}

var godaddyIssuerString = `/Country=US/Organization=The Go Daddy Group, Inc./OrganizationalUnit=Go Daddy Class 2 Certification Authority`
var godaddySubjectString = `/Country=US/Province=Arizona/Locality=Scottsdale/Organization=GoDaddy.com, Inc./OrganizationalUnit=http://certificates.godaddy.com/repository/CommonName=Go Daddy Secure Certification Authority/SerialNumber=07969287`

// Test marshal to JSON
// Also serves as a JSON format regression test.
func TestBundleMarshalJSON(t *testing.T) {
	b := newBundler(t)
	bundle, _ := b.BundleFromPEMorDER(GoDaddyIntermediateCert, nil, Optimal, "")
	bytes, err := json.Marshal(bundle)

	if err != nil {
		t.Fatal(err)
	}

	var obj bundleObject
	err = json.Unmarshal(bytes, &obj)
	if err != nil {
		t.Fatal(err)
	}

	if obj.Bundle == "" {
		t.Fatal("bundle is empty.")
	}
	if obj.Bundle != string(GoDaddyIntermediateCert) {
		t.Fatal("bundle is incorrect:", obj.Bundle)
	}

	if obj.Key != "" {
		t.Fatal("key is not empty:", obj.Key)
	}

	if obj.Root != string(GoDaddyRootCert) {
		t.Fatal("Root is not recovered")
	}

	if obj.Cert != string(GoDaddyIntermediateCert) {
		t.Fatal("Cert is not recovered")
	}

	if obj.KeyType != "2048-bit RSA" {
		t.Fatal("Incorrect key type:", obj.KeyType)
	}

	if obj.KeySize != 2048 {
		t.Fatal("Incorrect key size:", obj.KeySize)
	}

	if obj.Issuer != godaddyIssuerString {
		t.Fatal("Incorrect issuer:", obj.Issuer)
	}

	if obj.Subject != godaddySubjectString {
		t.Fatal("Incorrect subject:", obj.Subject)
	}

	if obj.Expires != "2026-11-16T01:54:37Z" {
		t.Fatal("Incorrect expiration time:", obj.Expires)
	}

	if len(obj.Hostnames) != 1 || obj.Hostnames[0] != "Go Daddy Secure Certification Authority" {
		t.Fatal("Incorrect hostnames:", obj.Hostnames)
	}

	if obj.OCSPSupport != true {
		t.Fatal("Incorrect OCSP support flag:", obj.OCSPSupport)
	}

	if obj.CRLSupport != true {
		t.Fatal("Incorrect CRL support flag:", obj.CRLSupport)
	}

	if len(obj.OCSP) != 1 || obj.OCSP[0] != `http://ocsp.godaddy.com` {
		t.Fatal("Incorrect ocsp server list:", obj.OCSP)
	}

	if obj.Signature != "SHA1WithRSA" {
		t.Fatal("Incorrect cert signature method:", obj.Signature)
	}
}

func TestBundleWithECDSAKeyMarshalJSON(t *testing.T) {
	b := newCustomizedBundlerFromFile(t, testCFSSLRootBundle, testCFSSLIntBundle, "")
	bundle, _ := b.BundleFromFile(leafECDSA256, leafKeyECDSA256, Optimal, "")
	jsonBytes, err := json.Marshal(bundle)

	if err != nil {
		t.Fatal(err)
	}

	var obj map[string]interface{}
	err = json.Unmarshal(jsonBytes, &obj)
	if err != nil {
		t.Fatal(err)
	}

	key := obj["key"].(string)
	keyBytes, _ := ioutil.ReadFile(leafKeyECDSA256)
	keyBytes = bytes.Trim(keyBytes, " \n")
	if key != string(keyBytes) {
		t.Fatal("key is not recovered.")
	}

	cert := obj["crt"].(string)
	certBytes, _ := ioutil.ReadFile(leafECDSA256)
	certBytes = bytes.Trim(certBytes, " \n")
	if cert != string(certBytes) {
		t.Fatal("cert is not recovered.")
	}

	keyType := obj["key_type"]
	if keyType != "256-bit ECDSA" {
		t.Fatal("Incorrect key type:", keyType)
	}

}

func TestBundleWithRSAKeyMarshalJSON(t *testing.T) {
	b := newCustomizedBundlerFromFile(t, testCFSSLRootBundle, testCFSSLIntBundle, "")
	bundle, _ := b.BundleFromFile(leafRSA2048, leafKeyRSA2048, Optimal, "")
	jsonBytes, err := json.Marshal(bundle)

	if err != nil {
		t.Fatal(err)
	}

	var obj map[string]interface{}
	err = json.Unmarshal(jsonBytes, &obj)
	if err != nil {
		t.Fatal(err)
	}

	key := obj["key"].(string)
	keyBytes, _ := ioutil.ReadFile(leafKeyRSA2048)
	keyBytes = bytes.Trim(keyBytes, " \n")
	if key != string(keyBytes) {
		t.Error("key is", key)
		t.Error("keyBytes is", string(keyBytes))
		t.Fatal("key is not recovered.")
	}

	cert := obj["crt"].(string)
	certBytes, _ := ioutil.ReadFile(leafRSA2048)
	certBytes = bytes.Trim(certBytes, " \n")
	if cert != string(certBytes) {
		t.Fatal("cert is not recovered.")
	}

	keyType := obj["key_type"]
	if keyType != "2048-bit RSA" {
		t.Fatal("Incorrect key type:", keyType)
	}

}

// Test marshal to JSON on hostnames
func TestBundleHostnamesMarshalJSON(t *testing.T) {
	b := newBundler(t)
	bundle, err := b.BundleFromRemote("www.cloudflare.com", "", Ubiquitous)
	if err != nil {
		t.Fatal(err)
	}
	hostnames, err := json.Marshal(bundle.Hostnames)
	if err != nil {
		t.Fatal(err)
	}
	expectedOne := []byte(`["www.cloudflare.com","cloudflare.com"]`)
	expectedTheOther := []byte(`["cloudflare.com","www.cloudflare.com"]`)
	if !bytes.Equal(hostnames, expectedOne) && !bytes.Equal(hostnames, expectedTheOther) {
		t.Fatal("Hostnames construction failed for cloudflare.com.", string(hostnames))
	}

	bundle, _ = b.BundleFromPEMorDER(GoDaddyIntermediateCert, nil, Optimal, "")
	expected := []byte(`["Go Daddy Secure Certification Authority"]`)
	hostnames, _ = json.Marshal(bundle.Hostnames)
	if !bytes.Equal(hostnames, expected) {
		t.Fatal("Hostnames construction failed for godaddy root cert.", string(hostnames))
	}

}

// Tests on verifying the rebundle flag and error code in Bundle.Status when rebundling.
func TestRebundleFromPEM(t *testing.T) {
	newBundler := newCustomizedBundlerFromFile(t, testCFSSLRootBundle, interL1, "")
	newBundle, err := newBundler.BundleFromPEMorDER(expiredBundlePEM, nil, Optimal, "")
	if err != nil {
		t.Fatalf("Re-bundle failed. %s", err.Error())
	}
	newChain := newBundle.Chain

	if len(newChain) != 2 {
		t.Fatalf("Expected bundle chain length is 2. Got %d.", len(newChain))
	}

	expiredChain, _ := helpers.ParseCertificatesPEM(expiredBundlePEM)
	for i, cert := range newChain {
		old := expiredChain[i]
		if i == 0 {
			if !bytes.Equal(old.Signature, cert.Signature) {
				t.Fatal("Leaf cert should be the same.")
			}
		} else {
			if bytes.Equal(old.Signature, cert.Signature) {
				t.Fatal("Intermediate cert should be different.")
			}
		}
	}
	// The status must be {Code: ExpiringBit is not set, IsRebundled:true, ExpiringSKIs:{}}
	if len(newBundle.Status.ExpiringSKIs) != 0 || !newBundle.Status.IsRebundled || newBundle.Status.Code&errors.BundleExpiringBit != 0 {
		t.Fatal("Rebundle Status is incorrect.")
	}

}

func TestRebundleExpiring(t *testing.T) {
	// make a policy that generate a cert expires in one hour.
	expiry := 1 * time.Hour
	policy := &config.Signing{
		Profiles: map[string]*config.SigningProfile{
			"expireIn1Hour": {
				Usage:  []string{"cert sign"},
				Expiry: expiry,
				CA:     true,
			},
		},
		Default: config.DefaultConfig(),
	}
	// Generate a intermediate cert that expires in one hour.
	expiringPEM := createInterCert(t, interL1CSR, policy, "expireIn1Hour")
	rootBundlePEM, _ := ioutil.ReadFile(testCFSSLRootBundle)

	// Use the expiring intermediate to initiate a bundler.
	bundler, err := NewBundlerFromPEM(rootBundlePEM, expiringPEM)
	if err != nil {
		t.Fatalf("bundle failed. %s", err.Error())
	}
	newBundle, err := bundler.BundleFromPEMorDER(expiredBundlePEM, nil, Optimal, "")
	if err != nil {
		t.Fatalf("Re-bundle failed. %s", err.Error())
	}
	// Check the bundle content.
	newChain := newBundle.Chain
	if len(newChain) != 2 {
		t.Fatalf("Expected bundle chain length is 2. Got %d.", len(newChain))
	}
	// The status must be {Code: ExpiringBit is set, IsRebundled:true, ExpiringSKIs:{"8860BA18A477B841041BD5EF7751C25B14BA203F"}}
	if len(newBundle.Status.ExpiringSKIs) != 1 || !newBundle.Status.IsRebundled || newBundle.Status.Code&errors.BundleExpiringBit == 0 {
		t.Fatal("Rebundle Status is incorrect.")
	}
	expectedSKI := "8860BA18A477B841041BD5EF7751C25B14BA203F"
	if newBundle.Status.ExpiringSKIs[0] != expectedSKI {
		t.Fatalf("Expected expiring cert SKI is %s, got %s\n", expectedSKI, newBundle.Status.ExpiringSKIs[0])
	}

}

// Test on verifying ubiquitous messaging in Bundle.Status.
func TestUbiquitousBundle(t *testing.T) {
	L1Cert := readCert(interL1)
	// Simulate the case that L1Cert is added to trust store by one platform but not yet in another.
	b := newCustomizedBundlerFromFile(t, testCFSSLRootBundle, testCFSSLIntBundle, "")
	b.RootPool.AddCert(L1Cert)
	// Prepare Platforms.
	platformA := ubiquity.Platform{Name: "MacroSoft", Weight: 100, HashAlgo: "SHA2", KeyAlgo: "ECDSA256", KeyStoreFile: testCFSSLRootBundle}
	platformA.ParseAndLoad()
	platformB := ubiquity.Platform{Name: "Godzilla", Weight: 100, HashAlgo: "SHA2", KeyAlgo: "ECDSA256", KeyStoreFile: testCFSSLRootBundle}
	platformB.ParseAndLoad()
	platformA.KeyStore.Add(L1Cert)
	ubiquity.Platforms = []ubiquity.Platform{platformA, platformB}

	// Optimal bundle algorithm will picks up the new root and shorten the chain.
	optimalBundle, err := b.BundleFromFile(leafECDSA256, "", Optimal, "")
	if err != nil {
		t.Fatal("Optimal bundle failed:", err)
	}
	if len(optimalBundle.Chain) != 2 {
		t.Fatal("Optimal bundle failed the chain length test. Chain length:", len(optimalBundle.Chain))
	}
	// The only trust platform is "Macrosoft".
	if len(optimalBundle.Status.Untrusted) != 1 {
		t.Fatal("Optimal bundle status has incorrect untrusted platforms", optimalBundle.Status.Untrusted)
	}
	checkUbiquityWarningAndCode(t, optimalBundle, true)

	// Ubiquitous bundle will remain the same.
	ubiquitousBundle, err := b.BundleFromFile(leafECDSA256, "", Ubiquitous, "")
	if err != nil {
		t.Fatal("Ubiquitous bundle failed")

	}
	if len(ubiquitousBundle.Chain) != 3 {
		t.Fatal("Ubiquitous bundle failed")
	}
	// Should be trusted by both platforms.
	if len(ubiquitousBundle.Status.Untrusted) != 0 {
		t.Fatal("Ubiquitous bundle status has incorrect untrusted platforms", len(ubiquitousBundle.Status.Untrusted))
	}
	checkUbiquityWarningAndCode(t, ubiquitousBundle, false)
}

func TestUbiquityBundleWithoutMetadata(t *testing.T) {
	b := newCustomizedBundlerFromFile(t, testCFSSLRootBundle, testCFSSLIntBundle, "")
	L1Cert := readCert(interL1)
	b.RootPool.AddCert(L1Cert)

	// Without platform info, ubiquitous bundling falls back to optimal bundling.
	ubiquity.Platforms = nil
	nuBundle, err := b.BundleFromFile(leafECDSA256, "", Ubiquitous, "")
	if err != nil {
		t.Fatal("Ubiquitous-fall-back-to-optimal bundle failed: ", err)

	}
	if len(nuBundle.Chain) != 2 {
		t.Fatal("Ubiquitous-fall-back-to-optimal bundle failed")
	}
	// Should be trusted by all (i.e. zero) platforms.
	if len(nuBundle.Status.Untrusted) != 0 {
		t.Fatal("Ubiquitous-fall-back-to-optimal bundle status has incorrect untrusted platforms", len(nuBundle.Status.Untrusted))
	}
	checkUbiquityWarningAndCode(t, nuBundle, true)
}

func checkUbiquityWarningAndCode(t *testing.T, bundle *Bundle, expected bool) {
	found := false
	for _, msg := range bundle.Status.Messages {
		if strings.Contains(msg, untrustedWarningStub) || strings.Contains(msg, ubiquityWarning) {
			found = true
		}
	}
	if found != expected {
		t.Fatal("Expected ubiquity warning: ", expected, " Found ubiquity warning:", found)
	}

	// check status code
	if expected && bundle.Status.Code&errors.BundleNotUbiquitousBit == 0 {
		t.Fatal("Bundle status doesn't set BundleNotUbiquitousBit :", bundle.Status.Code)
	}
}

// Regression test on bundle with all flavors:
// Ubiquitous bundle optimizes bundle length given the platform ubiquity is the same; Force bundle
// with return the same bundle; Optimal bundle always chooses shortest bundle length.
func TestForceBundle(t *testing.T) {
	// create a CA signer and signs a new intermediate with SHA-2
	caSigner := makeCASignerFromFile(testCAFile, testCAKeyFile, x509.SHA256WithRSA, t)
	interL1Bytes := signCSRFile(caSigner, interL1CSR, t)

	// create a inter L1 signer
	interL1KeyBytes, err := ioutil.ReadFile(interL1Key)
	if err != nil {
		t.Fatal(err)
	}

	interL1Signer := makeCASigner(interL1Bytes, interL1KeyBytes, x509.SHA256WithRSA, t)

	// sign a level 2 intermediate
	interL2Bytes := signCSRFile(interL1Signer, interL2CSR, t)

	// create a inter L2 signer
	interL2KeyBytes, err := ioutil.ReadFile(interL2Key)
	if err != nil {
		t.Fatal(err)
	}

	interL2Signer := makeCASigner(interL2Bytes, interL2KeyBytes, x509.ECDSAWithSHA256, t)

	// interL2 sign a leaf cert
	leafBytes := signCSRFile(interL2Signer, leafCSR, t)

	// create two platforms
	// both trust the CA cert and L1 intermediate
	caBytes, err := ioutil.ReadFile(testCAFile)
	if err != nil {
		t.Fatal(err)
	}

	ca, _ := helpers.ParseCertificatePEM(caBytes)
	interL1, _ := helpers.ParseCertificatePEM(interL1Bytes)
	platformA := ubiquity.Platform{
		Name:            "A",
		Weight:          100,
		KeyStore:        make(ubiquity.CertSet),
		HashUbiquity:    ubiquity.SHA2Ubiquity,
		KeyAlgoUbiquity: ubiquity.ECDSA521Ubiquity,
	}
	platformB := ubiquity.Platform{
		Name:            "B",
		Weight:          100,
		KeyStore:        make(ubiquity.CertSet),
		HashUbiquity:    ubiquity.SHA2Ubiquity,
		KeyAlgoUbiquity: ubiquity.ECDSA521Ubiquity,
	}

	platformA.KeyStore.Add(ca)
	platformA.KeyStore.Add(interL1)
	platformB.KeyStore.Add(ca)
	platformB.KeyStore.Add(interL1)
	ubiquity.Platforms = []ubiquity.Platform{platformA, platformB}

	caBundle := string(caBytes) + string(interL1Bytes)
	interBundle := string(interL2Bytes) + string(interL1Bytes)
	fullChain := string(leafBytes) + string(interL2Bytes) + string(interL1Bytes)

	// create bundler
	b, err := NewBundlerFromPEM([]byte(caBundle), []byte(interBundle))
	if err != nil {
		t.Fatal(err)
	}

	// The input PEM bundle is 3-cert chain.
	bundle, err := b.BundleFromPEMorDER([]byte(fullChain), nil, Force, "")
	if err != nil {
		t.Fatal("Force bundle failed:", err)
	}
	if len(bundle.Chain) != 3 {
		t.Fatal("Force bundle failed:")
	}
	if len(bundle.Status.Untrusted) != 0 {
		t.Fatal("Force bundle failed:")
	}

	// With ubiquity flavor, we should have a shorter chain, given L1 is ubiquitous trusted.
	bundle, err = b.BundleFromPEMorDER([]byte(fullChain), nil, Ubiquitous, "")
	if err != nil {
		t.Fatal("Ubiquitous bundle failed:", err)
	}
	if len(bundle.Chain) != 2 {
		t.Fatal("Ubiquitous bundle failed:")
	}
	if len(bundle.Status.Untrusted) != 0 {
		t.Fatal("Ubiquitous bundle failed:")
	}

	// With optimal flavor, we should have a shorter chain as well.
	bundle, err = b.BundleFromPEMorDER([]byte(fullChain), nil, Optimal, "")
	if err != nil {
		t.Fatal("Optimal bundle failed:", err)
	}
	if len(bundle.Chain) != 2 {
		t.Fatal("Optimal bundle failed:")
	}
	if len(bundle.Status.Untrusted) != 0 {
		t.Fatal("Optimal bundle failed:")
	}
}

func TestUpdateIntermediate(t *testing.T) {
	// create a CA signer and signs a new intermediate with SHA-2
	caSigner := makeCASignerFromFile(testCAFile, testCAKeyFile, x509.SHA256WithRSA, t)
	sha2InterBytes := signCSRFile(caSigner, interL1CSR, t)

	interKeyBytes, err := ioutil.ReadFile(interL1Key)
	if err != nil {
		t.Fatal(err)
	}

	// create a intermediate signer from intermediate cert/key
	sha2InterSigner := makeCASigner(sha2InterBytes, interKeyBytes, x509.SHA256WithRSA, t)
	// sign a leaf cert
	leafBytes := signCSRFile(sha2InterSigner, leafCSR, t)

	// read CA cert bytes
	caCertBytes, err := ioutil.ReadFile(testCAFile)
	if err != nil {
		t.Fatal(err)
	}
	// create a bundler with the test root CA and no intermediates
	b, err := NewBundlerFromPEM(caCertBytes, nil)
	if err != nil {
		t.Fatal(err)
	}

	// create a cert bundle: leaf + inter
	chainBytes := string(leafBytes) + string(sha2InterBytes)
	bundle, err := b.BundleFromPEMorDER([]byte(chainBytes), nil, Ubiquitous, "")
	if err != nil {
		t.Fatal("Valid bundle should be accepted. error:", err)
	}
	if bundle.Status.IsRebundled {
		t.Fatal("rebundle should never happen here", bundle.Status)
	}

	// Now bundle with the leaf cert
	bundle2, err := b.BundleFromPEMorDER(leafBytes, nil, Ubiquitous, "")
	if err != nil {
		t.Fatal("Valid bundle should be accepted. error:", err)
	}
	if !bundle2.Status.IsRebundled {
		t.Fatal("rebundle should happen here")
	}
}

func TestForceBundleNoFallback(t *testing.T) {
	// create a CA signer and signs a new intermediate with SHA-2
	caSigner := makeCASignerFromFile(testCAFile, testCAKeyFile, x509.SHA256WithRSA, t)
	sha2InterBytes := signCSRFile(caSigner, interL1CSR, t)

	interKeyBytes, err := ioutil.ReadFile(interL1Key)
	if err != nil {
		t.Fatal(err)
	}

	// create a intermediate signer from intermediate cert/key
	sha2InterSigner := makeCASigner(sha2InterBytes, interKeyBytes, x509.SHA256WithRSA, t)
	// sign a leaf cert
	leafBytes := signCSRFile(sha2InterSigner, leafCSR, t)

	// read CA cert bytes
	caCertBytes, err := ioutil.ReadFile(testCAFile)
	if err != nil {
		t.Fatal(err)
	}
	// create a bundler with the test root CA and the new intermediate
	b, err := NewBundlerFromPEM(caCertBytes, sha2InterBytes)
	if err != nil {
		t.Fatal(err)
	}

	// Now bundle with the leaf cert with Force
	bundle, err := b.BundleFromPEMorDER(leafBytes, nil, Force, "")
	if err != nil {
		t.Fatal("Valid bundle should be generated, error:", err)
	}

	// Force bundle fallback to creating a valid bundle
	if len(bundle.Chain) != 1 {
		t.Fatal("incorrect bundling")
	}
	if bundle.Status.IsRebundled {
		t.Fatal("rebundle should happen here")
	}

}

// Regression test: ubiquity bundle test with SHA2-homogeneous preference should not override root ubiquity.
func TestSHA2HomogeneityAgainstUbiquity(t *testing.T) {
	// create a CA signer and signs a new intermediate with SHA-1
	caSigner := makeCASignerFromFile(testCAFile, testCAKeyFile, x509.SHA1WithRSA, t)
	interL1Bytes := signCSRFile(caSigner, interL1CSR, t)

	// create a inter L1 signer
	interL1KeyBytes, err := ioutil.ReadFile(interL1Key)
	if err != nil {
		t.Fatal(err)
	}

	interL1Signer := makeCASigner(interL1Bytes, interL1KeyBytes, x509.SHA256WithRSA, t)

	// sign a level 2 intermediate
	interL2Bytes := signCSRFile(interL1Signer, interL2CSR, t)

	// create a inter L2 signer
	interL2KeyBytes, err := ioutil.ReadFile(interL2Key)
	if err != nil {
		t.Fatal(err)
	}

	interL2Signer := makeCASigner(interL2Bytes, interL2KeyBytes, x509.ECDSAWithSHA256, t)

	// interL2 sign a leaf cert
	leafBytes := signCSRFile(interL2Signer, leafCSR, t)

	// create two platforms
	// platform A trusts the CA cert and L1 intermediate
	// platform B trusts the CA cert
	caBytes, err := ioutil.ReadFile(testCAFile)
	if err != nil {
		t.Fatal(err)
	}

	ca, _ := helpers.ParseCertificatePEM(caBytes)
	interL1, _ := helpers.ParseCertificatePEM(interL1Bytes)
	platformA := ubiquity.Platform{
		Name:            "A",
		Weight:          100,
		KeyStore:        make(ubiquity.CertSet),
		HashUbiquity:    ubiquity.SHA2Ubiquity,
		KeyAlgoUbiquity: ubiquity.ECDSA521Ubiquity,
	}
	platformB := ubiquity.Platform{
		Name:            "B",
		Weight:          100,
		KeyStore:        make(ubiquity.CertSet),
		HashUbiquity:    ubiquity.SHA2Ubiquity,
		KeyAlgoUbiquity: ubiquity.ECDSA521Ubiquity,
	}

	platformA.KeyStore.Add(ca)
	platformA.KeyStore.Add(interL1)
	platformB.KeyStore.Add(ca)
	ubiquity.Platforms = []ubiquity.Platform{platformA, platformB}

	caBundle := string(caBytes) + string(interL1Bytes)
	interBundle := string(interL2Bytes) + string(interL1Bytes)
	fullChain := string(leafBytes) + string(interL2Bytes) + string(interL1Bytes)

	// create bundler
	b, err := NewBundlerFromPEM([]byte(caBundle), []byte(interBundle))
	if err != nil {
		t.Fatal(err)
	}

	// The input PEM bundle is 3-cert chain.
	bundle, err := b.BundleFromPEMorDER([]byte(fullChain), nil, Force, "")
	if err != nil {
		t.Fatal("Force bundle failed:", err)
	}
	if len(bundle.Chain) != 3 {
		t.Fatal("Force bundle failed:")
	}
	if len(bundle.Status.Untrusted) != 0 {
		t.Fatal("Force bundle failed:")
	}

	// With ubiquity flavor, we should not sacrifice trust store ubiquity and rebundle with a shorter chain
	// with SHA2 homogenity.
	bundle, err = b.BundleFromPEMorDER([]byte(fullChain), nil, Ubiquitous, "")
	if err != nil {
		t.Fatal("Ubiquitous bundle failed:", err)
	}
	if len(bundle.Chain) != 3 {
		t.Fatal("Ubiquitous bundle failed:")
	}
	if len(bundle.Status.Untrusted) != 0 {
		t.Fatal("Ubiquitous bundle failed:")
	}

	// With optimal flavor, we should have a shorter chain.
	bundle, err = b.BundleFromPEMorDER([]byte(fullChain), nil, Optimal, "")
	if err != nil {
		t.Fatal("Optimal bundle failed:", err)
	}
	if len(bundle.Chain) != 2 {
		t.Fatal("Optimal bundle failed:")
	}
	if len(bundle.Status.Untrusted) == 0 {
		t.Fatal("Optimal bundle failed:")
	}

}

func checkSHA2WarningAndCode(t *testing.T, bundle *Bundle, expected bool) {
	found := false
	for _, msg := range bundle.Status.Messages {
		if strings.Contains(msg, sha2Warning) {
			found = true
		}
	}
	if found != expected {
		t.Fatal("Expected ubiquity warning: ", expected, " Found ubiquity warning:", found)
	}
	// check status code
	if bundle.Status.Code&errors.BundleNotUbiquitousBit == 0 {
		t.Fatal("Bundle status code is incorrect:", bundle.Status.Code)
	}
}

func checkECDSAWarningAndCode(t *testing.T, bundle *Bundle, expected bool) {
	found := false
	for _, msg := range bundle.Status.Messages {
		if strings.Contains(msg, ecdsaWarning) {
			found = true
		}
	}
	if found != expected {
		t.Fatal("Expected ubiquity warning: ", expected, " Found ubiquity warning:", found)
	}
	// check status code
	if bundle.Status.Code&errors.BundleNotUbiquitousBit == 0 {
		t.Fatal("Bundle status code is incorrect:", bundle.Status.Code)
	}
}

// Regression test on SHA-2 Warning
// Riot Games once bundle a cert issued by DigiCert SHA2 High Assurance Server CA. The resulting
// bundle uses SHA-256 which is not supported in Windows XP SP2. We should present a warning
// on this.
func TestSHA2Warning(t *testing.T) {
	// create a CA signer and signs a new intermediate with SHA-2
	caSigner := makeCASignerFromFile(testCAFile, testCAKeyFile, x509.SHA256WithRSA, t)
	sha2InterBytes := signCSRFile(caSigner, interL1CSR, t)

	// read CA cert bytes
	caCertBytes, err := ioutil.ReadFile(testCAFile)
	if err != nil {
		t.Fatal(err)
	}

	// create a bundler with the test root CA and no intermediates
	b, err := NewBundlerFromPEM(caCertBytes, nil)
	if err != nil {
		t.Fatal(err)
	}

	optimalBundle, err := b.BundleFromPEMorDER(sha2InterBytes, nil, Optimal, "")
	if err != nil {
		t.Fatal("Optimal bundle failed:", err)
	}
	checkSHA2WarningAndCode(t, optimalBundle, true)

	// Ubiquitous bundle will include a 2nd intermediate CA.
	ubiquitousBundle, err := b.BundleFromPEMorDER(sha2InterBytes, nil, Ubiquitous, "")
	if err != nil {
		t.Fatal("Ubiquitous bundle failed")

	}
	checkSHA2WarningAndCode(t, ubiquitousBundle, true)
}

// Regression test on ECDSA Warning
// A test bundle that contains ECDSA384 and SHA-2. Expect ECDSA warning and SHA-2 warning.
func TestECDSAWarning(t *testing.T) {
	b := newCustomizedBundlerFromFile(t, testCAFile, interL1SHA1, "")

	optimalBundle, err := b.BundleFromFile(interL2SHA2, "", Optimal, "")
	if err != nil {
		t.Fatal("Optimal bundle failed:", err)
	}

	checkSHA2WarningAndCode(t, optimalBundle, true)
	checkECDSAWarningAndCode(t, optimalBundle, true)
}

// === Helper function block ===

// readCert read a PEM file and returns a cert.
func readCert(filename string) *x509.Certificate {
	bytes, _ := ioutil.ReadFile(filename)
	cert, _ := helpers.ParseCertificatePEM(bytes)
	return cert
}

// newBundler is a helper function that returns a new Bundler. If it fails to do so,
// it fails the test suite immediately.
func newBundler(t *testing.T) (b *Bundler) {
	b, err := NewBundler(testCaBundle, testIntCaBundle)
	if err != nil {
		t.Fatal(err)
	}
	return
}

// create a test intermediate cert in PEM
func createInterCert(t *testing.T, csrFile string, policy *config.Signing, profileName string) (certPEM []byte) {
	s, err := local.NewSignerFromFile(testCAFile, testCAKeyFile, policy)
	if err != nil {
		t.Fatal(err)
	}
	csr, err := ioutil.ReadFile(csrFile)
	if err != nil {
		t.Fatal(err)
	}
	req := signer.SignRequest{
		Hosts:   []string{"cloudflare-inter.com"},
		Request: string(csr),
		Profile: profileName,
		Label:   "",
	}

	certPEM, err = s.Sign(req)
	if err != nil {
		t.Fatal(err)
	}
	return

}

// newBundler creates bundler from byte slices of CA certs and intermediate certs in PEM format
func newBundlerFromPEM(t *testing.T, caBundlePEM, intBundlePEM []byte) (b *Bundler) {
	b, err := NewBundlerFromPEM(caBundlePEM, intBundlePEM)
	if err != nil {
		t.Fatal(err)
	}
	return
}

// newCustomizedBundleCreator is a helper function that returns a new Bundler
// takes specified CA bundle, intermediate bundle, and any additional intermdiate certs to generate a bundler.
func newCustomizedBundlerFromFile(t *testing.T, caBundle, intBundle, adhocInters string) (b *Bundler) {
	b, err := NewBundler(caBundle, intBundle)
	if err != nil {
		t.Fatal(err)
	}
	if adhocInters != "" {
		moreIntersPEM, err := ioutil.ReadFile(adhocInters)
		if err != nil {
			t.Fatalf("Read additional intermediates failed. %v",
				err)
		}
		intermediates, err := helpers.ParseCertificatesPEM(moreIntersPEM)
		if err != nil {
			t.Fatalf("Parsing additional intermediates failed. %s", err.Error())
		}
		for _, c := range intermediates {
			b.IntermediatePool.AddCert(c)
		}

	}
	return

}

// newBundlerWithoutInters is a helper function that returns a bundler with an empty
// intermediate cert pool. Such bundlers can help testing error handling in cert
// bundling.
func newBundlerWithoutInters(t *testing.T) (b *Bundler) {
	b = newBundler(t)
	// Re-assign an empty intermediate cert pool
	b.IntermediatePool = x509.NewCertPool()
	return
}

// newBundlerWithoutRoots is a helper function that returns a bundler with an empty
// root cert pool. Such bundlers can help testing error handling in cert
// bundling.
func newBundlerWithoutRoots(t *testing.T) (b *Bundler) {
	b = newBundler(t)
	// Re-assign an empty root cert pool
	b.RootPool = x509.NewCertPool()
	return
}

func newBundlerWithoutRootsAndInters(t *testing.T) *Bundler {
	b, err := NewBundler("", "")
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// A helper function that returns a errorCallback function which expects certain error content in
// an error message.
func ExpectErrorMessage(expectedErrorContent string) func(*testing.T, error) {
	return func(t *testing.T, err error) {
		if err == nil {
			t.Fatalf("Expected error has %s. Got nothing.", expectedErrorContent)
		} else if !strings.Contains(err.Error(), expectedErrorContent) {
			t.Fatalf("Expected error has %s. Got %s", expectedErrorContent, err.Error())
		}
	}
}

// A helper function that returns a errorCallback function which inspect error message for
// all expected messages.
func ExpectErrorMessages(expectedContents []string) func(*testing.T, error) {
	return func(t *testing.T, err error) {
		if err == nil {
			t.Fatalf("Expected error has %s. Got nothing.", expectedContents)
		} else {
			for _, expected := range expectedContents {
				if !strings.Contains(err.Error(), expected) {
					t.Fatalf("Expected error has %s. Got %s", expected, err.Error())
				}
			}
		}
	}
}

// A helper function that returns a bundle chain length checking function
func ExpectBundleLength(expectedLen int) func(*testing.T, *Bundle) {
	return func(t *testing.T, bundle *Bundle) {
		if bundle == nil {
			t.Fatalf("Cert bundle should have a chain of length %d. Got nil.",
				expectedLen)
		} else if len(bundle.Chain) != expectedLen {
			t.Fatalf("Cert bundle should have a chain of length %d. Got chain length %d.",
				expectedLen, len(bundle.Chain))
		}
	}
}

func TestBundlerWithEmptyRootInfo(t *testing.T) {
	b := newBundlerWithoutRootsAndInters(t)

	// "force" bundle should be ok
	bundle, err := b.BundleFromPEMorDER(GoDaddyIntermediateCert, nil, Force, "")
	if err != nil {
		t.Fatal(err)
	}
	checkBundleFunc := ExpectBundleLength(1)
	checkBundleFunc(t, bundle)

	// force non-verifying bundle should fail.
	_, err = b.BundleFromFile(badBundle, "", Force, "")
	if err == nil {
		t.Fatal("expected error. but no error occurred")
	}
	checkErrorFunc := ExpectErrorMessage("\"code\":1200")
	checkErrorFunc(t, err)

	// "optimal" and "ubiquitous" bundle should be ok
	bundle, err = b.BundleFromPEMorDER(GoDaddyIntermediateCert, nil, Ubiquitous, "")
	if err != nil {
		t.Fatal(err)
	}
	checkBundleFunc = ExpectBundleLength(1)
	checkBundleFunc(t, bundle)

	bundle, err = b.BundleFromPEMorDER(GoDaddyIntermediateCert, nil, Optimal, "")
	if err != nil {
		t.Fatal(err)
	}
	checkBundleFunc = ExpectBundleLength(1)
	checkBundleFunc(t, bundle)

	// bundle remote should be ok
	bundle, err = b.BundleFromRemote("www.google.com", "", Ubiquitous)
	if err != nil {
		t.Fatal(err)
	}
	checkBundleFunc = ExpectBundleLength(2)
	checkBundleFunc(t, bundle)
}
