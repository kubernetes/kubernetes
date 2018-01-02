package bundler

// This test file contains tests on checking Bundle.Status with SHA-1 deprecation warning.
import (
	"crypto/x509"
	"io/ioutil"
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
	sha1CA           = "testdata/ca.pem"
	sha1CAKey        = "testdata/ca.key"
	sha1Intermediate = "testdata/inter-L1-sha1.pem"
	sha2Intermediate = "testdata/inter-L1.pem"
	intermediateKey  = "testdata/inter-L1.key"
	intermediateCSR  = "testdata/inter-L1.csr"
	leafCSR          = "testdata/cfssl-leaf-ecdsa256.csr"
)

func TestChromeWarning(t *testing.T) {
	b := newCustomizedBundlerFromFile(t, sha1CA, sha1Intermediate, "")

	s, err := local.NewSignerFromFile(sha1Intermediate, intermediateKey, nil)
	if err != nil {
		t.Fatal(err)
	}

	csrBytes, err := ioutil.ReadFile(leafCSR)
	if err != nil {
		t.Fatal(err)
	}

	signingRequest := signer.SignRequest{Request: string(csrBytes)}

	certBytes, err := s.Sign(signingRequest)
	if err != nil {
		t.Fatal(err)
	}

	// Bundle a leaf cert with default 1 year expiration
	bundle, err := b.BundleFromPEMorDER(certBytes, nil, Ubiquitous, "")
	if err != nil {
		t.Fatal("bundling failed: ", err)
	}

	// should be not ubiquitous due to SHA2 and ECDSA support issues in legacy platforms
	if bundle.Status.Code&errors.BundleNotUbiquitousBit != errors.BundleNotUbiquitousBit {
		t.Fatal("Incorrect bundle status code. Bundle status code:", bundle.Status.Code)
	}

	fullChain := append(bundle.Chain, bundle.Root)
	sha1Msgs := ubiquity.SHA1DeprecationMessages(fullChain)
	// Since the new SHA-1 cert is expired after 2015, it definitely trigger Chrome's deprecation policies.
	if len(sha1Msgs) == 0 {
		t.Fatal("SHA1 Deprecation Message should not be empty")
	}
	// check SHA1 deprecation warnings
	var sha1MsgNotFound bool
	for _, sha1Msg := range sha1Msgs {
		foundMsg := false
		for _, message := range bundle.Status.Messages {
			if message == sha1Msg {
				foundMsg = true
			}
		}
		if !foundMsg {
			sha1MsgNotFound = true
			break
		}
	}
	if sha1MsgNotFound {
		t.Fatalf("Incorrect bundle status messages. Bundle status messages:%v, expected to contain: %v\n", bundle.Status.Messages, sha1Msgs)
	}

}

func TestSHA2Preferences(t *testing.T) {
	// create a CA signer and signs a new intermediate with SHA-1
	sha1CASigner := makeCASignerFromFile(sha1CA, sha1CAKey, x509.SHA1WithRSA, t)
	// create a CA signer and signs a new intermediate with SHA-2
	sha2CASigner := makeCASignerFromFile(sha1CA, sha1CAKey, x509.SHA256WithRSA, t)

	// sign two different intermediates
	sha1InterBytes := signCSRFile(sha1CASigner, intermediateCSR, t)
	sha2InterBytes := signCSRFile(sha2CASigner, intermediateCSR, t)

	interKeyBytes, err := ioutil.ReadFile(intermediateKey)
	if err != nil {
		t.Fatal(err)
	}

	// create a intermediate signer from SHA-1 intermediate cert/key
	sha2InterSigner := makeCASigner(sha1InterBytes, interKeyBytes, x509.SHA256WithRSA, t)
	// sign a leaf cert
	leafBytes := signCSRFile(sha2InterSigner, leafCSR, t)

	// create a bundler with SHA-1 and SHA-2 intermediate certs of same key.
	b := newCustomizedBundlerFromFile(t, sha1CA, sha1Intermediate, "")
	if err != nil {
		t.Fatal(err)
	}
	sha1Inter, _ := helpers.ParseCertificatePEM(sha1InterBytes)
	sha2Inter, _ := helpers.ParseCertificatePEM(sha2InterBytes)
	b.IntermediatePool.AddCert(sha1Inter)
	b.IntermediatePool.AddCert(sha2Inter)

	bundle, err := b.BundleFromPEMorDER(leafBytes, nil, Ubiquitous, "")
	if err != nil {
		t.Fatal("bundling failed: ", err)
	}

	if bundle.Chain[1].SignatureAlgorithm != x509.SHA256WithRSA {
		t.Fatal("ubiquity selection by SHA-2 homogenity failed.")
	}

}

func makeCASignerFromFile(certFile, keyFile string, sigAlgo x509.SignatureAlgorithm, t *testing.T) signer.Signer {
	certBytes, err := ioutil.ReadFile(certFile)
	if err != nil {
		t.Fatal(err)
	}

	keyBytes, err := ioutil.ReadFile(keyFile)
	if err != nil {
		t.Fatal(err)
	}

	return makeCASigner(certBytes, keyBytes, sigAlgo, t)

}

func makeCASigner(certBytes, keyBytes []byte, sigAlgo x509.SignatureAlgorithm, t *testing.T) signer.Signer {
	cert, err := helpers.ParseCertificatePEM(certBytes)
	if err != nil {
		t.Fatal(err)
	}

	key, err := helpers.ParsePrivateKeyPEM(keyBytes)
	if err != nil {
		t.Fatal(err)
	}

	defaultProfile := &config.SigningProfile{
		Usage:        []string{"cert sign"},
		CA:           true,
		Expiry:       time.Hour,
		ExpiryString: "1h",
	}
	policy := &config.Signing{
		Profiles: map[string]*config.SigningProfile{},
		Default:  defaultProfile,
	}
	s, err := local.NewSigner(key, cert, sigAlgo, policy)
	if err != nil {
		t.Fatal(err)
	}

	return s
}

func signCSRFile(s signer.Signer, csrFile string, t *testing.T) []byte {
	csrBytes, err := ioutil.ReadFile(csrFile)
	if err != nil {
		t.Fatal(err)
	}

	signingRequest := signer.SignRequest{Request: string(csrBytes)}
	certBytes, err := s.Sign(signingRequest)
	if err != nil {
		t.Fatal(err)
	}

	return certBytes
}
