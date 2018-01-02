package local

import (
	"bytes"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/asn1"
	"encoding/hex"
	"encoding/pem"
	"io/ioutil"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/cloudflare/cfssl/config"
	"github.com/cloudflare/cfssl/csr"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/log"
	"github.com/cloudflare/cfssl/signer"
)

const (
	fullSubjectCSR     = "testdata/test.csr"
	testCSR            = "testdata/ecdsa256.csr"
	testSANCSR         = "testdata/san_domain.csr"
	testCaFile         = "testdata/ca.pem"
	testCaKeyFile      = "testdata/ca_key.pem"
	testECDSACaFile    = "testdata/ecdsa256_ca.pem"
	testECDSACaKeyFile = "testdata/ecdsa256_ca_key.pem"
)

var expiry = 1 * time.Minute

// Start a signer with the testing RSA CA cert and key.
func newTestSigner(t *testing.T) (s *Signer) {
	s, err := NewSignerFromFile(testCaFile, testCaKeyFile, nil)
	if err != nil {
		t.Fatal(err)
	}
	return
}

func TestNewSignerFromFilePolicy(t *testing.T) {
	var CAConfig = &config.Config{
		Signing: &config.Signing{
			Profiles: map[string]*config.SigningProfile{
				"signature": {
					Usage:  []string{"digital signature"},
					Expiry: expiry,
				},
			},
			Default: &config.SigningProfile{
				Usage:        []string{"cert sign", "crl sign"},
				ExpiryString: "43800h",
				Expiry:       expiry,
				CA:           true,
			},
		},
	}
	_, err := NewSignerFromFile(testCaFile, testCaKeyFile, CAConfig.Signing)
	if err != nil {
		t.Fatal(err)
	}
}

func TestNewSignerFromFileInvalidPolicy(t *testing.T) {
	var invalidConfig = &config.Config{
		Signing: &config.Signing{
			Profiles: map[string]*config.SigningProfile{
				"invalid": {
					Usage:  []string{"wiretapping"},
					Expiry: expiry,
				},
				"empty": {},
			},
			Default: &config.SigningProfile{
				Usage:  []string{"digital signature"},
				Expiry: expiry,
			},
		},
	}
	_, err := NewSignerFromFile(testCaFile, testCaKeyFile, invalidConfig.Signing)
	if err == nil {
		t.Fatal(err)
	}

	if !strings.Contains(err.Error(), `"code":5200`) {
		t.Fatal(err)
	}
}

func TestNewSignerFromFileNoUsageInPolicy(t *testing.T) {
	var invalidConfig = &config.Config{
		Signing: &config.Signing{
			Profiles: map[string]*config.SigningProfile{
				"invalid": {
					Usage:  []string{},
					Expiry: expiry,
				},
				"empty": {},
			},
			Default: &config.SigningProfile{
				Usage:  []string{"digital signature"},
				Expiry: expiry,
			},
		},
	}
	_, err := NewSignerFromFile(testCaFile, testCaKeyFile, invalidConfig.Signing)
	if err == nil {
		t.Fatal("expect InvalidPolicy error")
	}

	if !strings.Contains(err.Error(), `"code":5200`) {
		t.Fatal(err)
	}
}

func TestNewSignerFromFileEdgeCases(t *testing.T) {

	res, err := NewSignerFromFile("nil", "nil", nil)
	if res != nil && err == nil {
		t.Fatal("Incorrect inputs failed to produce correct results")
	}

	res, err = NewSignerFromFile(testCaFile, "nil", nil)
	if res != nil && err == nil {
		t.Fatal("Incorrect inputs failed to produce correct results")
	}

	res, err = NewSignerFromFile("../../helpers/testdata/messedupcert.pem", "local.go", nil)
	if res != nil && err == nil {
		t.Fatal("Incorrect inputs failed to produce correct results")
	}

	res, err = NewSignerFromFile("../../helpers/testdata/cert.pem", "../../helpers/testdata/messed_up_priv_key.pem", nil)
	if res != nil && err == nil {
		t.Fatal("Incorrect inputs failed to produce correct results")
	}
}

// test the private method
func testSign(t *testing.T) {
	signer, err := NewSignerFromFile("testdata/ca.pem", "testdata/ca_key.pem", nil)
	if signer == nil || err != nil {
		t.Fatal("Failed to produce signer")
	}

	pem, _ := ioutil.ReadFile("../../helpers/testdata/cert.pem")
	cert, _ := helpers.ParseCertificatePEM(pem)

	badcert := *cert
	badcert.PublicKey = nil
	profl := config.SigningProfile{Usage: []string{"Certificates", "Rule"}}
	_, err = signer.sign(&badcert, &profl)

	if err == nil {
		t.Fatal("Improper input failed to raise an error")
	}

	// nil profile
	_, err = signer.sign(cert, &profl)
	if err == nil {
		t.Fatal("Nil profile failed to raise an error")
	}

	// empty profile
	_, err = signer.sign(cert, &config.SigningProfile{})
	if err == nil {
		t.Fatal("Empty profile failed to raise an error")
	}

	// empty expiry
	prof := signer.policy.Default
	prof.Expiry = 0
	_, err = signer.sign(cert, prof)
	if err != nil {
		t.Fatal("nil expiry raised an error")
	}

	// non empty urls
	prof = signer.policy.Default
	prof.CRL = "stuff"
	prof.OCSP = "stuff"
	prof.IssuerURL = []string{"stuff"}
	_, err = signer.sign(cert, prof)
	if err != nil {
		t.Fatal("non nil urls raised an error")
	}

	// nil ca
	nilca := *signer
	prof = signer.policy.Default
	prof.CA = false
	nilca.ca = nil
	_, err = nilca.sign(cert, prof)
	if err == nil {
		t.Fatal("nil ca with isca false raised an error")
	}
	prof.CA = true
	_, err = nilca.sign(cert, prof)
	if err != nil {
		t.Fatal("nil ca with CA true raised an error")
	}
}

func TestSign(t *testing.T) {
	testSign(t)
	s, err := NewSignerFromFile("testdata/ca.pem", "testdata/ca_key.pem", nil)
	if err != nil {
		t.Fatal("Failed to produce signer")
	}

	// test the empty request
	_, err = s.Sign(signer.SignRequest{})
	if err == nil {
		t.Fatalf("Empty request failed to produce an error")
	}

	// not a csr
	certPem, err := ioutil.ReadFile("../../helpers/testdata/cert.pem")
	if err != nil {
		t.Fatal(err)
	}

	// csr with ip as hostname
	pem, err := ioutil.ReadFile("testdata/ip.csr")
	if err != nil {
		t.Fatal(err)
	}

	// improper request
	validReq := signer.SignRequest{Hosts: signer.SplitHosts(testHostName), Request: string(certPem)}
	_, err = s.Sign(validReq)
	if err == nil {
		t.Fatal("A bad case failed to raise an error")
	}

	validReq = signer.SignRequest{Hosts: signer.SplitHosts("128.84.126.213"), Request: string(pem)}
	_, err = s.Sign(validReq)
	if err != nil {
		t.Fatal("A bad case failed to raise an error")
	}

	pem, err = ioutil.ReadFile("testdata/ex.csr")
	validReq = signer.SignRequest{
		Request: string(pem),
		Hosts:   []string{"example.com"},
	}
	s.Sign(validReq)
	if err != nil {
		t.Fatal("Failed to sign")
	}
}

func TestCertificate(t *testing.T) {
	s, err := NewSignerFromFile("testdata/ca.pem", "testdata/ca_key.pem", nil)
	if err != nil {
		t.Fatal(err)
	}

	c, err := s.Certificate("", "")
	if !reflect.DeepEqual(*c, *s.ca) || err != nil {
		t.Fatal("Certificate() producing incorrect results")
	}
}

func TestPolicy(t *testing.T) {
	s, err := NewSignerFromFile("testdata/ca.pem", "testdata/ca_key.pem", nil)
	if err != nil {
		t.Fatal(err)
	}

	sgn := config.Signing{}

	s.SetPolicy(&sgn)
	if s.Policy() != &sgn {
		t.Fatal("Policy is malfunctioning")
	}
}

func newCustomSigner(t *testing.T, testCaFile, testCaKeyFile string) (s *Signer) {
	s, err := NewSignerFromFile(testCaFile, testCaKeyFile, nil)
	if err != nil {
		t.Fatal(err)
	}
	return
}

func TestNewSignerFromFile(t *testing.T) {
	newTestSigner(t)
}

const (
	testHostName = "localhost"
)

func testSignFile(t *testing.T, certFile string) ([]byte, error) {
	s := newTestSigner(t)

	pem, err := ioutil.ReadFile(certFile)
	if err != nil {
		t.Fatal(err)
	}

	return s.Sign(signer.SignRequest{Hosts: signer.SplitHosts(testHostName), Request: string(pem)})
}

type csrTest struct {
	file    string
	keyAlgo string
	keyLen  int
	// Error checking function
	errorCallback func(*testing.T, error)
}

// A helper function that returns a errorCallback function which expects an error.
func ExpectError() func(*testing.T, error) {
	return func(t *testing.T, err error) {
		if err == nil {
			t.Fatal("Expected error. Got nothing.")
		}
	}
}

var csrTests = []csrTest{
	{
		file:          "testdata/rsa2048.csr",
		keyAlgo:       "rsa",
		keyLen:        2048,
		errorCallback: nil,
	},
	{
		file:          "testdata/rsa3072.csr",
		keyAlgo:       "rsa",
		keyLen:        3072,
		errorCallback: nil,
	},
	{
		file:          "testdata/rsa4096.csr",
		keyAlgo:       "rsa",
		keyLen:        4096,
		errorCallback: nil,
	},
	{
		file:          "testdata/ecdsa256.csr",
		keyAlgo:       "ecdsa",
		keyLen:        256,
		errorCallback: nil,
	},
	{
		file:          "testdata/ecdsa384.csr",
		keyAlgo:       "ecdsa",
		keyLen:        384,
		errorCallback: nil,
	},
	{
		file:          "testdata/ecdsa521.csr",
		keyAlgo:       "ecdsa",
		keyLen:        521,
		errorCallback: nil,
	},
}

func TestSignCSRs(t *testing.T) {
	s := newTestSigner(t)
	hostname := "cloudflare.com"
	for _, test := range csrTests {
		csr, err := ioutil.ReadFile(test.file)
		if err != nil {
			t.Fatal("CSR loading error:", err)
		}
		// It is possible to use different SHA2 algorithm with RSA CA key.
		rsaSigAlgos := []x509.SignatureAlgorithm{x509.SHA1WithRSA, x509.SHA256WithRSA, x509.SHA384WithRSA, x509.SHA512WithRSA}
		for _, sigAlgo := range rsaSigAlgos {
			s.sigAlgo = sigAlgo
			certBytes, err := s.Sign(signer.SignRequest{Hosts: signer.SplitHosts(hostname), Request: string(csr)})
			if test.errorCallback != nil {
				test.errorCallback(t, err)
			} else {
				if err != nil {
					t.Fatalf("Expected no error. Got %s. Param %s %d", err.Error(), test.keyAlgo, test.keyLen)
				}
				cert, _ := helpers.ParseCertificatePEM(certBytes)
				if cert.SignatureAlgorithm != s.SigAlgo() {
					t.Fatal("Cert Signature Algorithm does not match the issuer.")
				}
			}
		}
	}
}

func TestECDSASigner(t *testing.T) {
	s := newCustomSigner(t, testECDSACaFile, testECDSACaKeyFile)
	hostname := "cloudflare.com"
	for _, test := range csrTests {
		csr, err := ioutil.ReadFile(test.file)
		if err != nil {
			t.Fatal("CSR loading error:", err)
		}
		// Try all ECDSA SignatureAlgorithm
		SigAlgos := []x509.SignatureAlgorithm{x509.ECDSAWithSHA1, x509.ECDSAWithSHA256, x509.ECDSAWithSHA384, x509.ECDSAWithSHA512}
		for _, sigAlgo := range SigAlgos {
			s.sigAlgo = sigAlgo
			certBytes, err := s.Sign(signer.SignRequest{Hosts: signer.SplitHosts(hostname), Request: string(csr)})
			if test.errorCallback != nil {
				test.errorCallback(t, err)
			} else {
				if err != nil {
					t.Fatalf("Expected no error. Got %s. Param %s %d", err.Error(), test.keyAlgo, test.keyLen)
				}
				cert, _ := helpers.ParseCertificatePEM(certBytes)
				if cert.SignatureAlgorithm != s.SigAlgo() {
					t.Fatal("Cert Signature Algorithm does not match the issuer.")
				}
			}
		}
	}
}

const (
	ecdsaInterCSR = "testdata/ecdsa256-inter.csr"
	ecdsaInterKey = "testdata/ecdsa256-inter.key"
	rsaInterCSR   = "testdata/rsa2048-inter.csr"
	rsaInterKey   = "testdata/rsa2048-inter.key"
)

func TestCAIssuing(t *testing.T) {
	var caCerts = []string{testCaFile, testECDSACaFile}
	var caKeys = []string{testCaKeyFile, testECDSACaKeyFile}
	var interCSRs = []string{ecdsaInterCSR, rsaInterCSR}
	var interKeys = []string{ecdsaInterKey, rsaInterKey}
	var CAPolicy = &config.Signing{
		Default: &config.SigningProfile{
			Usage:        []string{"cert sign", "crl sign"},
			ExpiryString: "1h",
			Expiry:       1 * time.Hour,
			CA:           true,
		},
	}
	var hostname = "cloudflare-inter.com"
	// Each RSA or ECDSA root CA issues two intermediate CAs (one ECDSA and one RSA).
	// For each intermediate CA, use it to issue additional RSA and ECDSA intermediate CSRs.
	for i, caFile := range caCerts {
		caKeyFile := caKeys[i]
		s := newCustomSigner(t, caFile, caKeyFile)
		s.policy = CAPolicy
		for j, csr := range interCSRs {
			csrBytes, _ := ioutil.ReadFile(csr)
			certBytes, err := s.Sign(signer.SignRequest{Hosts: signer.SplitHosts(hostname), Request: string(csrBytes)})
			if err != nil {
				t.Fatal(err)
			}
			interCert, err := helpers.ParseCertificatePEM(certBytes)
			if err != nil {
				t.Fatal(err)
			}
			keyBytes, _ := ioutil.ReadFile(interKeys[j])
			interKey, _ := helpers.ParsePrivateKeyPEM(keyBytes)
			interSigner := &Signer{
				ca:      interCert,
				priv:    interKey,
				policy:  CAPolicy,
				sigAlgo: signer.DefaultSigAlgo(interKey),
			}
			for _, anotherCSR := range interCSRs {
				anotherCSRBytes, _ := ioutil.ReadFile(anotherCSR)
				bytes, err := interSigner.Sign(
					signer.SignRequest{
						Hosts:   signer.SplitHosts(hostname),
						Request: string(anotherCSRBytes),
					})
				if err != nil {
					t.Fatal(err)
				}
				cert, err := helpers.ParseCertificatePEM(bytes)
				if err != nil {
					t.Fatal(err)
				}
				if cert.SignatureAlgorithm != interSigner.SigAlgo() {
					t.Fatal("Cert Signature Algorithm does not match the issuer.")
				}
			}
		}
	}

}

func TestPopulateSubjectFromCSR(t *testing.T) {
	// a subject with all its fields full.
	fullSubject := &signer.Subject{
		CN: "CN",
		Names: []csr.Name{
			{
				C:  "C",
				ST: "ST",
				L:  "L",
				O:  "O",
				OU: "OU",
			},
		},
		SerialNumber: "deadbeef",
	}

	fullName := pkix.Name{
		CommonName:         "CommonName",
		Country:            []string{"Country"},
		Province:           []string{"Province"},
		Organization:       []string{"Organization"},
		OrganizationalUnit: []string{"OrganizationalUnit"},
		SerialNumber:       "SerialNumber",
	}

	noCN := *fullSubject
	noCN.CN = ""
	name := PopulateSubjectFromCSR(&noCN, fullName)
	if name.CommonName != "CommonName" {
		t.Fatal("Failed to replace empty common name")
	}

	noC := *fullSubject
	noC.Names[0].C = ""
	name = PopulateSubjectFromCSR(&noC, fullName)
	if !reflect.DeepEqual(name.Country, fullName.Country) {
		t.Fatal("Failed to replace empty country")
	}

	noL := *fullSubject
	noL.Names[0].L = ""
	name = PopulateSubjectFromCSR(&noL, fullName)
	if !reflect.DeepEqual(name.Locality, fullName.Locality) {
		t.Fatal("Failed to replace empty locality")
	}

	noO := *fullSubject
	noO.Names[0].O = ""
	name = PopulateSubjectFromCSR(&noO, fullName)
	if !reflect.DeepEqual(name.Organization, fullName.Organization) {
		t.Fatal("Failed to replace empty organization")
	}

	noOU := *fullSubject
	noOU.Names[0].OU = ""
	name = PopulateSubjectFromCSR(&noOU, fullName)
	if !reflect.DeepEqual(name.OrganizationalUnit, fullName.OrganizationalUnit) {
		t.Fatal("Failed to replace empty organizational unit")
	}

	noSerial := *fullSubject
	noSerial.SerialNumber = ""
	name = PopulateSubjectFromCSR(&noSerial, fullName)
	if name.SerialNumber != fullName.SerialNumber {
		t.Fatalf("Failed to replace empty serial number: want %#v, got %#v", fullName.SerialNumber, name.SerialNumber)
	}

}
func TestOverrideSubject(t *testing.T) {
	csrPEM, err := ioutil.ReadFile(fullSubjectCSR)
	if err != nil {
		t.Fatalf("%v", err)
	}

	req := &signer.Subject{
		Names: []csr.Name{
			{O: "example.net"},
		},
	}

	s := newCustomSigner(t, testECDSACaFile, testECDSACaKeyFile)

	request := signer.SignRequest{
		Hosts:   []string{"127.0.0.1", "localhost", "xyz@example.com"},
		Request: string(csrPEM),
		Subject: req,
	}

	certPEM, err := s.Sign(request)

	if err != nil {
		t.Fatalf("%v", err)
	}

	cert, err := helpers.ParseCertificatePEM(certPEM)
	if err != nil {
		t.Fatalf("%v", err)
	}

	block, _ := pem.Decode(csrPEM)
	template, err := x509.ParseCertificateRequest(block.Bytes)
	if err != nil {
		t.Fatal(err.Error())
	}
	if cert.Subject.Organization[0] != "example.net" {
		t.Fatalf("Failed to override subject: want example.net but have %s", cert.Subject.Organization[0])
	}

	if cert.Subject.Country[0] != template.Subject.Country[0] {
		t.Fatal("Failed to override Country")
	}

	if cert.Subject.Locality[0] != template.Subject.Locality[0] {
		t.Fatal("Failed to override Locality")
	}

	if cert.Subject.Organization[0] == template.Subject.Organization[0] {
		t.Fatal("Shouldn't have overrode Organization")
	}

	if cert.Subject.OrganizationalUnit[0] != template.Subject.OrganizationalUnit[0] {
		t.Fatal("Failed to override OrganizationalUnit")
	}

	log.Info("Overrode subject info")
}

func TestOverwriteHosts(t *testing.T) {
	for _, csrFile := range []string{testCSR, testSANCSR} {
		csrPEM, err := ioutil.ReadFile(csrFile)
		if err != nil {
			t.Fatal(err)
		}

		csrDER, _ := pem.Decode([]byte(csrPEM))
		if err != nil {
			t.Fatal(err)
		}

		csr, err := x509.ParseCertificateRequest(csrDER.Bytes)
		if err != nil {
			t.Fatal(err)
		}

		csrHosts := csr.DNSNames
		for _, ip := range csr.IPAddresses {
			csrHosts = append(csrHosts, ip.String())
		}
		sort.Strings(csrHosts)

		s := newCustomSigner(t, testECDSACaFile, testECDSACaKeyFile)

		for _, hosts := range [][]string{
			nil,
			{},
			{"127.0.0.1", "localhost", "xyz@example.com"},
		} {
			request := signer.SignRequest{
				Hosts:   hosts,
				Request: string(csrPEM),
				Subject: nil,
			}
			certPEM, err := s.Sign(request)

			if err != nil {
				t.Fatalf("%v", err)
			}

			cert, err := helpers.ParseCertificatePEM(certPEM)
			if err != nil {
				t.Fatalf("%v", err)
			}

			// get the hosts, and add the ips and email addresses
			certHosts := cert.DNSNames
			for _, ip := range cert.IPAddresses {
				certHosts = append(certHosts, ip.String())
			}

			for _, email := range cert.EmailAddresses {
				certHosts = append(certHosts, email)
			}

			// compare the sorted host lists
			sort.Strings(certHosts)
			sort.Strings(request.Hosts)
			if len(request.Hosts) > 0 && !reflect.DeepEqual(certHosts, request.Hosts) {
				t.Fatalf("Hosts not the same. cert hosts: %v, expected: %v", certHosts, request.Hosts)
			}

			if request.Hosts == nil && !reflect.DeepEqual(certHosts, csrHosts) {
				t.Fatalf("Hosts not the same. cert hosts: %v, expected csr hosts: %v", certHosts, csrHosts)
			}

			if request.Hosts != nil && len(request.Hosts) == 0 && len(certHosts) != 0 {
				t.Fatalf("Hosts not the same. cert hosts: %v, expected: %v", certHosts, request.Hosts)
			}
		}
	}

}

func expectOneValueOf(t *testing.T, s []string, e, n string) {
	if len(s) != 1 {
		t.Fatalf("Expected %s to have a single value, but it has %d values", n, len(s))
	}

	if s[0] != e {
		t.Fatalf("Expected %s to be '%s', but it is '%s'", n, e, s[0])
	}
}

func expectEmpty(t *testing.T, s []string, n string) {
	if len(s) != 0 {
		t.Fatalf("Expected no values in %s, but have %d values: %v", n, len(s), s)
	}
}

func TestNoWhitelistSign(t *testing.T) {
	csrPEM, err := ioutil.ReadFile(fullSubjectCSR)
	if err != nil {
		t.Fatalf("%v", err)
	}

	req := &signer.Subject{
		Names: []csr.Name{
			{O: "sam certificate authority"},
		},
		CN: "localhost",
	}

	s := newCustomSigner(t, testECDSACaFile, testECDSACaKeyFile)
	// No policy CSR whitelist: the normal set of CSR fields get passed through to
	// certificate.
	s.policy = &config.Signing{
		Default: &config.SigningProfile{
			Usage:        []string{"cert sign", "crl sign"},
			ExpiryString: "1h",
			Expiry:       1 * time.Hour,
			CA:           true,
		},
	}

	request := signer.SignRequest{
		Hosts:   []string{"127.0.0.1", "localhost"},
		Request: string(csrPEM),
		Subject: req,
	}

	certPEM, err := s.Sign(request)
	if err != nil {
		t.Fatalf("%v", err)
	}

	cert, err := helpers.ParseCertificatePEM(certPEM)
	if err != nil {
		t.Fatalf("%v", err)
	}

	name := cert.Subject
	if name.CommonName != "localhost" {
		t.Fatalf("Expected certificate common name to be 'localhost' but have '%v'", name.CommonName)
	}

	// CSR has: Subject: C=US, O=CloudFlare, OU=WWW, L=Ithaca, ST=New York
	// Expect all to be passed through.
	expectOneValueOf(t, name.Organization, "sam certificate authority", "O")
	expectOneValueOf(t, name.OrganizationalUnit, "WWW", "OU")
	expectOneValueOf(t, name.Province, "New York", "ST")
	expectOneValueOf(t, name.Locality, "Ithaca", "L")
	expectOneValueOf(t, name.Country, "US", "C")
}

func TestWhitelistSign(t *testing.T) {
	csrPEM, err := ioutil.ReadFile(fullSubjectCSR)
	if err != nil {
		t.Fatalf("%v", err)
	}

	req := &signer.Subject{
		Names: []csr.Name{
			{O: "sam certificate authority"},
		},
	}

	s := newCustomSigner(t, testECDSACaFile, testECDSACaKeyFile)
	// Whitelist only key-related fields. Subject, DNSNames, etc shouldn't get
	// passed through from CSR.
	s.policy = &config.Signing{
		Default: &config.SigningProfile{
			Usage:        []string{"cert sign", "crl sign"},
			ExpiryString: "1h",
			Expiry:       1 * time.Hour,
			CA:           true,
			CSRWhitelist: &config.CSRWhitelist{
				PublicKey:          true,
				PublicKeyAlgorithm: true,
				SignatureAlgorithm: true,
			},
		},
	}

	request := signer.SignRequest{
		Hosts:   []string{"127.0.0.1", "localhost"},
		Request: string(csrPEM),
		Subject: req,
	}

	certPEM, err := s.Sign(request)
	if err != nil {
		t.Fatalf("%v", err)
	}

	cert, err := helpers.ParseCertificatePEM(certPEM)
	if err != nil {
		t.Fatalf("%v", err)
	}

	name := cert.Subject
	if name.CommonName != "" {
		t.Fatalf("Expected empty certificate common name under policy without "+
			"Subject whitelist, got %v", name.CommonName)
	}
	// O is provided by the signing API request, not the CSR, so it's allowed to
	// be copied into the certificate.
	expectOneValueOf(t, name.Organization, "sam certificate authority", "O")
	expectEmpty(t, name.OrganizationalUnit, "OU")
	expectEmpty(t, name.Province, "ST")
	expectEmpty(t, name.Locality, "L")
	expectEmpty(t, name.Country, "C")
	if cert.PublicKeyAlgorithm != x509.RSA {
		t.Fatalf("Expected public key algorithm to be RSA")
	}

	// Signature algorithm is allowed to be copied from CSR, but is overridden by
	// DefaultSigAlgo.
	if cert.SignatureAlgorithm != x509.ECDSAWithSHA256 {
		t.Fatalf("Expected public key algorithm to be ECDSAWithSHA256, got %v",
			cert.SignatureAlgorithm)
	}
}

func TestNameWhitelistSign(t *testing.T) {
	csrPEM, err := ioutil.ReadFile(fullSubjectCSR)
	if err != nil {
		t.Fatalf("%v", err)
	}

	subInvalid := &signer.Subject{
		CN: "localhost.com",
	}
	subValid := &signer.Subject{
		CN: "1lab41.cf",
	}

	wl := regexp.MustCompile("^1[a-z]*[0-9]*\\.cf$")

	s := newCustomSigner(t, testECDSACaFile, testECDSACaKeyFile)
	// Whitelist only key-related fields. Subject, DNSNames, etc shouldn't get
	// passed through from CSR.
	s.policy = &config.Signing{
		Default: &config.SigningProfile{
			Usage:         []string{"cert sign", "crl sign"},
			ExpiryString:  "1h",
			Expiry:        1 * time.Hour,
			CA:            true,
			NameWhitelist: wl,
		},
	}

	request := signer.SignRequest{
		Hosts:   []string{"127.0.0.1", "1machine23.cf"},
		Request: string(csrPEM),
	}

	_, err = s.Sign(request)
	if err != nil {
		t.Fatalf("%v", err)
	}

	request = signer.SignRequest{
		Hosts:   []string{"invalid.cf", "1machine23.cf"},
		Request: string(csrPEM),
	}

	_, err = s.Sign(request)
	if err == nil {
		t.Fatalf("expected a policy error")
	}

	request = signer.SignRequest{
		Hosts:   []string{"1machine23.cf"},
		Request: string(csrPEM),
		Subject: subInvalid,
	}

	_, err = s.Sign(request)
	if err == nil {
		t.Fatalf("expected a policy error")
	}

	request = signer.SignRequest{
		Hosts:   []string{"1machine23.cf"},
		Request: string(csrPEM),
		Subject: subValid,
	}

	_, err = s.Sign(request)
	if err != nil {
		t.Fatalf("%v", err)
	}

}

func TestExtensionSign(t *testing.T) {
	csrPEM, err := ioutil.ReadFile(testCSR)
	if err != nil {
		t.Fatalf("%v", err)
	}

	s := newCustomSigner(t, testECDSACaFile, testECDSACaKeyFile)

	// By default, no extensions should be allowed
	request := signer.SignRequest{
		Request: string(csrPEM),
		Extensions: []signer.Extension{
			{ID: config.OID(asn1.ObjectIdentifier{1, 2, 3, 4})},
		},
	}

	_, err = s.Sign(request)
	if err == nil {
		t.Fatalf("expected a policy error")
	}

	// Whitelist a specific extension.  The extension with OID 1.2.3.4 should be
	// allowed through, but the one with OID 1.2.3.5 should not.
	s.policy = &config.Signing{
		Default: &config.SigningProfile{
			Usage:              []string{"cert sign", "crl sign"},
			ExpiryString:       "1h",
			Expiry:             1 * time.Hour,
			CA:                 true,
			ExtensionWhitelist: map[string]bool{"1.2.3.4": true},
		},
	}

	// Test that a forbidden extension triggers a sign error
	request = signer.SignRequest{
		Request: string(csrPEM),
		Extensions: []signer.Extension{
			{ID: config.OID(asn1.ObjectIdentifier{1, 2, 3, 5})},
		},
	}

	_, err = s.Sign(request)
	if err == nil {
		t.Fatalf("expected a policy error")
	}

	extValue := []byte{0x05, 0x00}
	extValueHex := hex.EncodeToString(extValue)

	// Test that an allowed extension makes it through
	request = signer.SignRequest{
		Request: string(csrPEM),
		Extensions: []signer.Extension{
			{
				ID:       config.OID(asn1.ObjectIdentifier{1, 2, 3, 4}),
				Critical: false,
				Value:    extValueHex,
			},
		},
	}

	certPEM, err := s.Sign(request)
	if err != nil {
		t.Fatalf("%v", err)
	}

	cert, err := helpers.ParseCertificatePEM(certPEM)
	if err != nil {
		t.Fatalf("%v", err)
	}

	foundAllowed := false
	for _, ext := range cert.Extensions {
		if ext.Id.String() == "1.2.3.4" {
			foundAllowed = true

			if ext.Critical {
				t.Fatalf("Extensions should not be marked critical")
			}

			if !bytes.Equal(extValue, ext.Value) {
				t.Fatalf("Extension has wrong value: %s != %s", hex.EncodeToString(ext.Value), extValueHex)
			}
		}
	}
	if !foundAllowed {
		t.Fatalf("Custom extension not included in the certificate")
	}
}

func TestCTFailure(t *testing.T) {
	var config = &config.Signing{
		Default: &config.SigningProfile{
			Expiry:       helpers.OneYear,
			CA:           true,
			Usage:        []string{"signing", "key encipherment", "server auth", "client auth"},
			ExpiryString: "8760h",
			CTLogServers: []string{"https://ct.googleapis.com/pilot"},
		},
	}
	testSigner, err := NewSignerFromFile(testCaFile, testCaKeyFile, config)
	if err != nil {
		t.Fatalf("%v", err)
	}
	var pem []byte
	pem, err = ioutil.ReadFile("testdata/ex.csr")
	if err != nil {
		t.Fatalf("%v", err)
	}
	validReq := signer.SignRequest{
		Request: string(pem),
		Hosts:   []string{"example.com"},
	}
	_, err = testSigner.Sign(validReq)

	// This should fail because our CA cert is not trusted by Google's pilot log
	if err == nil {
		t.Fatal("Expected CT log submission failure")
	}
}
