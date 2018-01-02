package ubiquity

import (
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"testing"
	"time"

	"github.com/cloudflare/cfssl/helpers"
)

const (
	rsa1024    = "testdata/rsa1024sha1.pem"
	rsa2048    = "testdata/rsa2048sha2.pem"
	rsa3072    = "testdata/rsa3072sha2.pem"
	rsa4096    = "testdata/rsa4096sha2.pem"
	ecdsa256   = "testdata/ecdsa256sha2.pem"
	ecdsa384   = "testdata/ecdsa384sha2.pem"
	ecdsa521   = "testdata/ecdsa521sha2.pem"
	caMetadata = "testdata/ca.pem.metadata"
)

var rsa1024Cert, rsa2048Cert, rsa3072Cert, rsa4096Cert, ecdsa256Cert, ecdsa384Cert, ecdsa521Cert *x509.Certificate

func readCert(filename string) *x509.Certificate {
	bytes, _ := ioutil.ReadFile(filename)
	cert, _ := helpers.ParseCertificatePEM(bytes)
	return cert
}

func init() {
	rsa1024Cert = readCert(rsa1024)
	rsa2048Cert = readCert(rsa2048)
	rsa3072Cert = readCert(rsa3072)
	rsa4096Cert = readCert(rsa4096)
	ecdsa256Cert = readCert(ecdsa256)
	ecdsa384Cert = readCert(ecdsa384)
	ecdsa521Cert = readCert(ecdsa521)

}

func TestCertHashPriority(t *testing.T) {
	if hashPriority(rsa1024Cert) > hashPriority(rsa2048Cert) {
		t.Fatal("Incorrect hash priority")
	}
	if hashPriority(rsa2048Cert) > hashPriority(rsa3072Cert) {
		t.Fatal("Incorrect hash priority")
	}
	if hashPriority(rsa3072Cert) > hashPriority(rsa4096Cert) {
		t.Fatal("Incorrect hash priority")
	}
	if hashPriority(rsa4096Cert) > hashPriority(ecdsa256Cert) {
		t.Fatal("Incorrect hash priority")
	}
	if hashPriority(ecdsa256Cert) > hashPriority(ecdsa384Cert) {
		t.Fatal("Incorrect hash priority")
	}
	if hashPriority(ecdsa384Cert) > hashPriority(ecdsa256Cert) {
		t.Fatal("Incorrect hash priority")
	}
}

func TestCertKeyAlgoPriority(t *testing.T) {
	if keyAlgoPriority(rsa2048Cert) > keyAlgoPriority(rsa3072Cert) {
		t.Fatal("Incorrect hash priority")
	}
	if keyAlgoPriority(rsa3072Cert) > keyAlgoPriority(rsa4096Cert) {
		t.Fatal("Incorrect hash priority")
	}
	if keyAlgoPriority(rsa4096Cert) > keyAlgoPriority(ecdsa256Cert) {
		t.Fatal("Incorrect hash priority")
	}
	if keyAlgoPriority(ecdsa256Cert) > keyAlgoPriority(ecdsa384Cert) {
		t.Fatal("Incorrect hash priority")
	}
	if keyAlgoPriority(ecdsa384Cert) > keyAlgoPriority(ecdsa521Cert) {
		t.Fatal("Incorrect hash priority")
	}
}
func TestChainHashPriority(t *testing.T) {
	var chain []*x509.Certificate
	var p int
	chain = []*x509.Certificate{rsa2048Cert, rsa3072Cert}
	p = HashPriority(chain)
	if p != (hashPriority(rsa2048Cert)+hashPriority(rsa3072Cert))/2 {
		t.Fatal("Incorrect chain hash priority")
	}
}

func TestChainKeyAlgoPriority(t *testing.T) {
	var chain []*x509.Certificate
	var p int
	chain = []*x509.Certificate{rsa2048Cert, rsa3072Cert}
	p = KeyAlgoPriority(chain)
	if p != (keyAlgoPriority(rsa2048Cert)+keyAlgoPriority(rsa3072Cert))/2 {
		t.Fatal("Incorrect chain key algo priority")
	}
}
func TestCertHashUbiquity(t *testing.T) {
	if hashUbiquity(rsa2048Cert) != SHA2Ubiquity {
		t.Fatal("incorrect hash ubiquity")
	}
	if hashUbiquity(rsa3072Cert) != SHA2Ubiquity {
		t.Fatal("incorrect hash ubiquity")
	}
	if hashUbiquity(rsa4096Cert) != SHA2Ubiquity {
		t.Fatal("incorrect hash ubiquity")
	}
	if hashUbiquity(rsa2048Cert) < hashUbiquity(rsa3072Cert) {
		t.Fatal("incorrect hash ubiquity")
	}
	if hashUbiquity(rsa3072Cert) < hashUbiquity(rsa4096Cert) {
		t.Fatal("Incorrect hash ubiquity")
	}
	if hashUbiquity(rsa4096Cert) < hashUbiquity(ecdsa256Cert) {
		t.Fatal("Incorrect hash ubiquity")
	}
	if hashUbiquity(ecdsa256Cert) < hashUbiquity(ecdsa384Cert) {
		t.Fatal("Incorrect hash ubiquity")
	}
	if hashUbiquity(ecdsa384Cert) < hashUbiquity(ecdsa256Cert) {
		t.Fatal("Incorrect hash ubiquity")
	}
}

func TestCertKeyAlgoUbiquity(t *testing.T) {
	if keyAlgoUbiquity(rsa2048Cert) != RSAUbiquity {
		t.Fatal("incorrect hash ubiquity")
	}
	if keyAlgoUbiquity(rsa3072Cert) != RSAUbiquity {
		t.Fatal("incorrect hash ubiquity")
	}
	if keyAlgoUbiquity(rsa4096Cert) != RSAUbiquity {
		t.Fatal("incorrect hash ubiquity")
	}
	if keyAlgoUbiquity(ecdsa256Cert) != ECDSA256Ubiquity {
		t.Fatal("incorrect hash ubiquity")
	}
	if keyAlgoUbiquity(ecdsa384Cert) != ECDSA384Ubiquity {
		t.Fatal("incorrect hash ubiquity")
	}
	if keyAlgoUbiquity(ecdsa521Cert) != ECDSA521Ubiquity {
		t.Fatal("incorrect hash ubiquity")
	}
	if keyAlgoUbiquity(rsa2048Cert) < keyAlgoUbiquity(rsa3072Cert) {
		t.Fatal("incorrect hash ubiquity")
	}
	if keyAlgoUbiquity(rsa3072Cert) < keyAlgoUbiquity(rsa4096Cert) {
		t.Fatal("Incorrect hash ubiquity")
	}
	if keyAlgoUbiquity(rsa4096Cert) < keyAlgoUbiquity(ecdsa256Cert) {
		t.Fatal("Incorrect hash ubiquity")
	}
	if keyAlgoUbiquity(ecdsa256Cert) < keyAlgoUbiquity(ecdsa384Cert) {
		t.Fatal("Incorrect hash ubiquity")
	}
	if keyAlgoUbiquity(ecdsa384Cert) < keyAlgoUbiquity(ecdsa256Cert) {
		t.Fatal("Incorrect hash ubiquity")
	}
}

func TestChainHashUbiquity(t *testing.T) {
	chain := []*x509.Certificate{rsa1024Cert, rsa2048Cert}
	if ChainHashUbiquity(chain) != hashUbiquity(rsa2048Cert) {
		t.Fatal("Incorrect chain hash ubiquity")
	}
}

func TestChainKeyAlgoUbiquity(t *testing.T) {
	chain := []*x509.Certificate{rsa1024Cert, rsa2048Cert}
	if ChainKeyAlgoUbiquity(chain) != keyAlgoUbiquity(rsa2048Cert) {
		t.Fatal("Incorrect chain hash ubiquity")
	}
	chain = []*x509.Certificate{ecdsa256Cert, rsa2048Cert}
	if ChainKeyAlgoUbiquity(chain) != keyAlgoUbiquity(ecdsa256Cert) {
		t.Fatal("Incorrect chain hash ubiquity")
	}

}

func TestChainExpiryUbiquity(t *testing.T) {
	// rsa1024Cert expires at year 2024
	// rsa2048Cert expires at year 2019
	// ecdsa256Cert expires at year 2019
	chain1 := []*x509.Certificate{ecdsa256Cert, rsa2048Cert}
	chain2 := []*x509.Certificate{ecdsa256Cert, rsa1024Cert}

	// CompareExpiryUbiquity should return > 0 because chain1
	// has a better expiry ubiquity than chain2.
	if CompareExpiryUbiquity(chain1, chain2) <= 0 {
		t.Fatal("Incorrect chain expiry ubiquity")
	}

	// CompareExpiryUbiquity should return < 0 because chain1 has
	// a better expiry ubiquity than chain2.
	if CompareExpiryUbiquity(chain2, chain1) >= 0 {
		t.Fatal("Incorrect chain expiry ubiquity")
	}

	if CompareExpiryUbiquity(chain1, chain1) != 0 {
		t.Fatal("Incorrect chain expiry ubiquity")
	}
}

func TestCompareChainExpiry(t *testing.T) {
	// rsa1024Cert expires at 2024
	// rsa2048Cert expires at 2019
	// ecdsa256Cert expires at 2019
	// both chain expires at year 2019.
	chain1 := []*x509.Certificate{ecdsa256Cert, rsa2048Cert}
	chain2 := []*x509.Certificate{ecdsa256Cert, rsa1024Cert}
	if CompareChainExpiry(chain1, chain2) != 0 {
		t.Fatal("Incorrect chain expiry")
	}

	if CompareExpiryUbiquity(chain1, chain1) != 0 {
		t.Fatal("Incorrect chain expiry")
	}
}

func TestCompareChainLength(t *testing.T) {
	chain1 := []*x509.Certificate{ecdsa256Cert, rsa2048Cert}
	chain2 := []*x509.Certificate{rsa1024Cert}
	chain3 := []*x509.Certificate{rsa2048Cert}
	// longer chain is ranked lower
	if CompareChainLength(chain1, chain2) >= 0 {
		t.Fatal("Incorrect chain length comparison")
	}

	if CompareChainLength(chain2, chain3) != 0 {
		t.Fatal("Incorrect chain length comparison")
	}
}

func TestPlatformKeyStoreUbiquity(t *testing.T) {
	cert1 := rsa1024Cert
	cert2 := rsa2048Cert
	cert3 := ecdsa256Cert
	// load Platforms with test data
	// "Macrosoft" has all three certs.
	// "Godzilla" has two certs, cert1 and cert2.
	// "Pinapple" has cert1.
	// "Colorful" has no key store data, default to trust any cert
	// All platforms support the same crypto suite.
	platformA := Platform{Name: "MacroSoft", Weight: 100, HashAlgo: "SHA2", KeyAlgo: "ECDSA256", KeyStoreFile: "testdata/macrosoft.pem"}
	platformB := Platform{Name: "Godzilla", Weight: 100, HashAlgo: "SHA2", KeyAlgo: "ECDSA256", KeyStoreFile: "testdata/godzilla.pem"}
	platformC := Platform{Name: "Pineapple", Weight: 100, HashAlgo: "SHA2", KeyAlgo: "ECDSA256", KeyStoreFile: "testdata/pineapple.pem"}
	platformD := Platform{Name: "Colorful", Weight: 100, HashAlgo: "SHA2", KeyAlgo: "ECDSA256", KeyStoreFile: ""}
	platformA.ParseAndLoad()
	platformB.ParseAndLoad()
	platformC.ParseAndLoad()
	platformD.ParseAndLoad()
	Platforms = []Platform{platformA, platformB, platformC, platformD}
	// chain1 with root cert1 (RSA1024, SHA1), has the largest platform coverage.
	// chain2 with root cert2 (RSA2048, SHA2), has the second largest coverage.
	// chain3 with root cert3 (ECDSA256, SHA2), has the least coverage.
	chain1 := []*x509.Certificate{cert1}
	chain2 := []*x509.Certificate{cert1, cert2}
	chain3 := []*x509.Certificate{cert1, cert2, cert3}
	if CrossPlatformUbiquity(chain1) < CrossPlatformUbiquity(chain2) {
		t.Fatal("Incorrect cross platform ubiquity")
	}
	if CrossPlatformUbiquity(chain2) < CrossPlatformUbiquity(chain3) {
		t.Fatal("Incorrect cross platform ubiquity")
	}

	if ComparePlatformUbiquity(chain1, chain2) < 0 {
		t.Fatal("Incorrect cross platform ubiquity")
	}

	if ComparePlatformUbiquity(chain2, chain3) < 0 {
		t.Fatal("Incorrect cross platform ubiquity")
	}

	// test UntrustedPlatforms()
	u1 := UntrustedPlatforms(cert1)
	if len(u1) != 0 {
		t.Fatal("Incorrect UntrustedPlatforms")
	}
	u2 := UntrustedPlatforms(cert2)
	if len(u2) != 1 {
		t.Fatal("Incorrect UntrustedPlatforms")
	}
	u3 := UntrustedPlatforms(cert3)
	if len(u3) != 2 {
		t.Fatal("Incorrect UntrustedPlatforms")
	}

}

func TestEmptyPlatformList(t *testing.T) {
	Platforms = []Platform{}
	cert := rsa1024Cert
	chain := []*x509.Certificate{cert}
	if CrossPlatformUbiquity(chain) != 0 {
		t.Fatal("Incorrect cross platform ubiquity when Platforms is empty")
	}
	// test UntrustedPlatforms()
	u1 := UntrustedPlatforms(cert)
	if len(u1) != 0 {
		t.Fatal("Incorrect UntrustedPlatforms when Platforms is empty")
	}
}

func TestLoadPlatforms(t *testing.T) {
	err := LoadPlatforms(caMetadata)
	if err != nil {
		t.Fatal(err)
	}
}

func TestPlatformCryptoUbiquity(t *testing.T) {
	cert1 := rsa1024Cert
	cert2 := rsa2048Cert
	cert3 := ecdsa256Cert
	// load Platforms with test data
	// All platforms have the same trust store but are with various crypto suite.
	platformA := Platform{Name: "TinySoft", Weight: 100, HashAlgo: "SHA1", KeyAlgo: "RSA", KeyStoreFile: "testdata/macrosoft.pem"}
	platformB := Platform{Name: "SmallSoft", Weight: 100, HashAlgo: "SHA2", KeyAlgo: "RSA", KeyStoreFile: "testdata/macrosoft.pem"}
	platformC := Platform{Name: "LargeSoft", Weight: 100, HashAlgo: "SHA2", KeyAlgo: "ECDSA256", KeyStoreFile: "testdata/macrosoft.pem"}
	platformD := Platform{Name: "MediumSoft", Weight: 100, HashAlgo: "SHA2", KeyAlgo: "ECDSA384", KeyStoreFile: "testdata/macrosoft.pem"}
	platformA.ParseAndLoad()
	platformB.ParseAndLoad()
	platformC.ParseAndLoad()
	platformD.ParseAndLoad()
	Platforms = []Platform{platformA, platformB, platformC}
	// chain1 with root cert1 (RSA1024, SHA1), has the largest platform coverage.
	// chain2 with root cert2 (RSA2048, SHA2), has the second largest coverage.
	// chain3 with root cert3 (ECDSA256, SHA2), has the least coverage.
	chain1 := []*x509.Certificate{cert1}
	chain2 := []*x509.Certificate{cert1, cert2}
	chain3 := []*x509.Certificate{cert1, cert2, cert3}
	if CrossPlatformUbiquity(chain1) < CrossPlatformUbiquity(chain2) {
		t.Fatal("Incorrect cross platform ubiquity")
	}
	if CrossPlatformUbiquity(chain2) < CrossPlatformUbiquity(chain3) {
		t.Fatal("Incorrect cross platform ubiquity")
	}

	if ComparePlatformUbiquity(chain1, chain2) < 0 {
		t.Fatal("Incorrect cross platform ubiquity")
	}

	if ComparePlatformUbiquity(chain1, chain2) < 0 {
		t.Fatal("Incorrect cross platform ubiquity")
	}
}

func TestSHA2Homogeneity(t *testing.T) {
	// root-only chain is always SHA2-Homogeneous.
	chain0 := []*x509.Certificate{rsa1024Cert}
	if SHA2Homogeneity(chain0) != 1 {
		t.Fatal("SHA2Homogeneity(chain0) != 1")
	}

	chain1 := []*x509.Certificate{rsa1024Cert, rsa2048Cert, rsa1024Cert}
	if SHA2Homogeneity(chain1) != 0 {
		t.Fatal("SHA2Homogeneity(chain1) != 0")
	}

	chain2 := []*x509.Certificate{rsa2048Cert, rsa2048Cert, rsa1024Cert}
	if SHA2Homogeneity(chain2) != 1 {
		t.Fatal("SHA2Homogeneity(chain2) != 1")
	}

	chain3 := []*x509.Certificate{ecdsa256Cert, rsa2048Cert, rsa1024Cert}
	if SHA2Homogeneity(chain3) != 1 {
		t.Fatal("SHA2Homogeneity(chain3) != 1")
	}

	chain4 := []*x509.Certificate{ecdsa256Cert, ecdsa384Cert, rsa1024Cert}
	if SHA2Homogeneity(chain4) != 1 {
		t.Fatal("SHA2Homogeneity(chain4) != 1")
	}
}

func TestCompareSHA2Homogeneity(t *testing.T) {
	chain1 := []*x509.Certificate{rsa1024Cert, rsa2048Cert, rsa1024Cert}
	chain2 := []*x509.Certificate{rsa2048Cert, rsa2048Cert, rsa1024Cert}
	chain3 := []*x509.Certificate{ecdsa256Cert, rsa2048Cert, rsa1024Cert}
	chain4 := []*x509.Certificate{ecdsa256Cert, ecdsa384Cert, rsa1024Cert}
	if CompareSHA2Homogeneity(chain1, chain2) >= 0 {
		t.Fatal("CompareSHA2Homogeneity(chain1, chain2) >= 0")
	}

	if CompareSHA2Homogeneity(chain1, chain3) >= 0 {
		t.Fatal("CompareSHA2Homogeneity(chain1, chain3) >= 0")
	}

	if CompareSHA2Homogeneity(chain1, chain4) >= 0 {
		t.Fatal("CompareSHA2Homogeneity(chain1, chain4) >= 0")
	}

	if CompareSHA2Homogeneity(chain2, chain3) != 0 || CompareSHA2Homogeneity(chain3, chain4) != 0 {
		t.Fatal("CompareSHA2Homogeneity failed.")
	}
}

func TestFilterTrivial(t *testing.T) {
	var chain []*x509.Certificate
	var chains [][]*x509.Certificate
	ret := Filter(chains, CompareChainHashPriority)
	if len(ret) != 0 {
		t.Fatal("Incorrect filtering")
	}

	chain = []*x509.Certificate{rsa2048Cert}
	chains = [][]*x509.Certificate{chain}

	ret = Filter(chains, CompareChainHashPriority)
	if len(ret) != 1 {
		t.Fatal("Incorrect filtering")
	}
}

func TestFilterChainHashPriority(t *testing.T) {
	var chain1, chain2 []*x509.Certificate
	chain1 = []*x509.Certificate{rsa2048Cert}  // SHA256
	chain2 = []*x509.Certificate{ecdsa384Cert} // SHA384
	// SHA256 <= SHA384
	if CompareChainHashPriority(chain1, chain2) > 0 {
		t.Fatal("Incorrect chain hash priority comparison")
	}
	chains := [][]*x509.Certificate{chain2, chain1}
	ret := Filter(chains, CompareChainHashPriority)

	// check there is no reordering
	if ret[0][0] != ecdsa384Cert {
		t.Fatal("Incorrect chain hash priority filtering")
	}

}

func TestFilterChainKeyAlgoPriority(t *testing.T) {
	var chain1, chain2 []*x509.Certificate
	chain1 = []*x509.Certificate{rsa2048Cert}  // RSA
	chain2 = []*x509.Certificate{ecdsa384Cert} // ECDSA
	// RSA <= ECDSA
	if CompareChainKeyAlgoPriority(chain1, chain2) >= 0 {
		t.Fatal("Incorrect chain key algo priority comparison")
	}
	chains := [][]*x509.Certificate{chain1, chain2}
	ret := Filter(chains, CompareChainKeyAlgoPriority)

	// check there is reordering
	if ret[0][0] != ecdsa384Cert {
		t.Fatal("Incorrect chain key algo priority filtering")
	}
}

func TestFilterChainCipherSuite(t *testing.T) {
	var chain1, chain2 []*x509.Certificate
	chain1 = []*x509.Certificate{rsa2048Cert}
	chain2 = []*x509.Certificate{ecdsa384Cert}
	// RSA2048 < ECDSA384
	if CompareChainCryptoSuite(chain1, chain2) >= 0 {
		t.Fatal("Incorrect chain key algo priority comparison")
	}
	chains := [][]*x509.Certificate{chain1, chain2}
	ret := Filter(chains, CompareChainCryptoSuite)

	// check there is reordering
	if ret[0][0] != ecdsa384Cert {
		t.Fatal("Incorrect chain key algo priority filtering")
	}
}

func TestFilterChainHashUbiquity(t *testing.T) {
	var chain1, chain2 []*x509.Certificate
	chain1 = []*x509.Certificate{rsa2048Cert}  // SHA256
	chain2 = []*x509.Certificate{ecdsa384Cert} // SHA384
	// SHA256 == SHA384
	if CompareChainHashUbiquity(chain1, chain2) != 0 {
		t.Fatal("Incorrect chain hash priority comparison")
	}
	chains := [][]*x509.Certificate{chain2, chain1}
	ret := Filter(chains, CompareChainHashUbiquity)

	// check there is no reordering
	if ret[0][0] != ecdsa384Cert {
		t.Fatal("Incorrect chain hash priority filtering")
	}
}

func TestFilterChainKeyAlgoUbiquity(t *testing.T) {
	var chain1, chain2 []*x509.Certificate
	chain1 = []*x509.Certificate{rsa2048Cert}  // RSA
	chain2 = []*x509.Certificate{ecdsa384Cert} // ECDSA
	// RSA >= ECDSA
	if CompareChainKeyAlgoUbiquity(chain1, chain2) < 0 {
		t.Fatal("Incorrect chain key algo priority comparison")
	}
	chains := [][]*x509.Certificate{chain1, chain2}
	ret := Filter(chains, CompareChainKeyAlgoUbiquity)

	// check there is no reordering
	if ret[0][0] != rsa2048Cert {
		t.Fatal("Incorrect chain key algo priority filtering")
	}
}

func TestFlagBySHA1DeprecationPolicy(t *testing.T) {
	cert1 := rsa1024Cert
	cert2 := rsa2048Cert
	Jan1st2014 := time.Date(2014, time.January, 1, 0, 0, 0, 0, time.UTC)
	Jan1st2100 := time.Date(2100, time.January, 1, 0, 0, 0, 0, time.UTC)
	policy1 := SHA1DeprecationPolicy{
		Description:    "SHA1 should be gone years ago",
		ExpiryDeadline: Jan1st2014,
	}
	policy2 := SHA1DeprecationPolicy{
		Description:    "SHA1 is perfect for another century",
		ExpiryDeadline: Jan1st2100,
	}
	policy3 := SHA1DeprecationPolicy{
		Description:    "effectively one century later, reject SHA1 expires on 2014",
		EffectiveDate:  Jan1st2100,
		ExpiryDeadline: Jan1st2014,
	}
	policy4 := SHA1DeprecationPolicy{
		Description:     "no more new SHA1 cert",
		NeverIssueAfter: Jan1st2014,
	}
	// chain1 is accepted univerally. It's not flagged because root cert is not subject to SHA1 deprecation.
	chain1 := []*x509.Certificate{cert1}
	if policy1.Flag(chain1) || policy2.Flag(chain1) || policy3.Flag(chain1) || policy4.Flag(chain1) {
		t.Fatal("Incorrect SHA1 deprecation")
	}

	// chain2 is accepted by policy2 and policy3. It's flagged by policy1 and policy4
	chain2 := []*x509.Certificate{cert1, cert1}
	if !policy1.Flag(chain2) || policy2.Flag(chain2) || policy3.Flag(chain2) || !policy4.Flag(chain2) {
		t.Fatal("Incorrect SHA1 deprecation")
	}

	// chain3 is accepted by universally since the leaf cert and the intermediate are signed by SHA-256
	chain3 := []*x509.Certificate{cert2, cert2, cert1}
	if policy1.Flag(chain3) || policy2.Flag(chain3) || policy3.Flag(chain3) || policy4.Flag(chain3) {
		t.Fatal("Incorrect SHA1 deprecation")
	}
}

func TestSHA1DeprecationMessages(t *testing.T) {
	cert1 := rsa1024Cert
	cert2 := rsa2048Cert
	chain1 := []*x509.Certificate{cert1}
	chain2 := []*x509.Certificate{cert1, cert1}
	chain3 := []*x509.Certificate{cert2, cert1, cert1}
	chain4 := []*x509.Certificate{cert2, cert2, cert1}
	messages := []string{}

	Jan1st2014 := time.Date(2014, time.January, 1, 0, 0, 0, 0, time.UTC)
	Jan1st2100 := time.Date(2100, time.January, 1, 0, 0, 0, 0, time.UTC)
	policy1 := SHA1DeprecationPolicy{
		Platform:       "Browser A",
		Description:    "minor warning",
		Severity:       Low,
		ExpiryDeadline: Jan1st2014,
	}
	policy2 := SHA1DeprecationPolicy{
		Platform:       "Browser A",
		Description:    "minor warning",
		Severity:       Medium,
		ExpiryDeadline: Jan1st2014,
	}
	policy3 := SHA1DeprecationPolicy{
		Platform:        "Browser B",
		Description:     "reject",
		Severity:        High,
		NeverIssueAfter: Jan1st2014,
	}
	policy4 := SHA1DeprecationPolicy{
		Platform:        "Browser C",
		Description:     "reject but not now",
		Severity:        High,
		NeverIssueAfter: Jan1st2014,
		EffectiveDate:   Jan1st2100,
	}

	// The only policy has severity low
	SHA1DeprecationPolicys = []SHA1DeprecationPolicy{policy1}
	messages = SHA1DeprecationMessages(chain1)
	// chain1 with only root is not subject to deprecation
	if len(messages) != 0 {
		t.Fatal("Incorrect SHA1 deprecation reporting")
	}
	// chain2 has SHA-1 leaf cert, subject to deprecation
	messages = SHA1DeprecationMessages(chain2)
	if len(messages) != 1 {
		t.Fatal("Incorrect SHA1 deprecation reporting")
	}
	// chain3 has SHA-1 intermediate cert, subject to deprecation
	messages = SHA1DeprecationMessages(chain3)
	if len(messages) != 1 {
		t.Fatal("Incorrect SHA1 deprecation reporting")
	}
	// chain4 has no SHA-1 leaf or intermediate, not subject to deprecation
	messages = SHA1DeprecationMessages(chain4)
	if len(messages) != 0 {
		t.Fatal("Incorrect SHA1 deprecation reporting")
	}

	// A second policy that has higher severity , so it should takes effect and override lower one.
	SHA1DeprecationPolicys = []SHA1DeprecationPolicy{policy1, policy2}
	// chain1 only has root cert, not subject to deprecation policy
	messages = SHA1DeprecationMessages(chain1)
	if len(messages) != 0 {
		t.Fatal("Incorrect SHA1 deprecation reporting")
	}
	// chain2 has a SHA-1 leaf cert, will have a message from policy2
	messages = SHA1DeprecationMessages(chain2)
	if len(messages) != 1 ||
		messages[0] != fmt.Sprintf("%s %s due to SHA-1 deprecation", policy2.Platform, policy2.Description) {
		t.Fatal("Incorrect SHA1 deprecation reporting")
	}
	// chain3 has a SHA-1 intermediate cert, will have a message from policy2
	messages = SHA1DeprecationMessages(chain3)
	if len(messages) != 1 ||
		messages[0] != fmt.Sprintf("%s %s due to SHA-1 deprecation", policy2.Platform, policy2.Description) {
		t.Fatal("Incorrect SHA1 deprecation reporting")
	}
	// chain4 is not subject to any deprecation policy
	messages = SHA1DeprecationMessages(chain4)
	if len(messages) != 0 {
		t.Fatal("Incorrect SHA1 deprecation reporting")
	}

	// Add two policies. One tests for newly issued leaf certificate after a deadline, the other is the same,
	// but takes effect at the 22nd century.
	SHA1DeprecationPolicys = []SHA1DeprecationPolicy{policy1, policy2, policy3, policy4}
	// chain1 only has root cert, not subject to any deprecation policy
	messages = SHA1DeprecationMessages(chain1)
	if len(messages) != 0 {
		t.Fatal("Incorrect SHA1 deprecation reporting")
	}
	// chain2 now is flagged by two policies: policy2 and policy3
	messages = SHA1DeprecationMessages(chain2)
	if len(messages) != 2 {
		t.Fatal("Incorrect SHA1 deprecation reporting")
	}
	// chain3 is not flagged by policy3 but policy2
	messages = SHA1DeprecationMessages(chain3)
	if len(messages) != 1 ||
		messages[0] != fmt.Sprintf("%s %s due to SHA-1 deprecation", policy2.Platform, policy2.Description) {
		t.Fatal("Incorrect SHA1 deprecation reporting")
	}
	// chain4 is not subject to any deprecation policy
	messages = SHA1DeprecationMessages(chain4)
	if len(messages) != 0 {
		t.Fatal("Incorrect SHA1 deprecation reporting")
	}
}
