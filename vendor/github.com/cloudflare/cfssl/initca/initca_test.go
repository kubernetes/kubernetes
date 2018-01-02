package initca

import (
	"crypto/ecdsa"
	"crypto/rsa"
	"io/ioutil"
	"strings"
	"testing"
	"time"

	"github.com/cloudflare/cfssl/config"
	"github.com/cloudflare/cfssl/csr"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/signer"
	"github.com/cloudflare/cfssl/signer/local"
)

var validKeyParams = []csr.BasicKeyRequest{
	{"rsa", 2048},
	{"rsa", 3072},
	{"rsa", 4096},
	{"ecdsa", 256},
	{"ecdsa", 384},
	{"ecdsa", 521},
}

var csrFiles = []string{
	"testdata/rsa2048.csr",
	"testdata/rsa3072.csr",
	"testdata/rsa4096.csr",
	"testdata/ecdsa256.csr",
	"testdata/ecdsa384.csr",
	"testdata/ecdsa521.csr",
}

var testRSACAFile = "testdata/5min-rsa.pem"
var testRSACAKeyFile = "testdata/5min-rsa-key.pem"
var testECDSACAFile = "testdata/5min-ecdsa.pem"
var testECDSACAKeyFile = "testdata/5min-ecdsa-key.pem"

var invalidCryptoParams = []csr.BasicKeyRequest{
	// Weak Key
	{"rsa", 1024},
	// Bad param
	{"rsaCrypto", 2048},
	{"ecdsa", 2000},
}

func TestInitCA(t *testing.T) {
	var req *csr.CertificateRequest
	hostname := "cloudflare.com"
	for _, param := range validKeyParams {
		req = &csr.CertificateRequest{
			Names: []csr.Name{
				{
					C:  "US",
					ST: "California",
					L:  "San Francisco",
					O:  "CloudFlare",
					OU: "Systems Engineering",
				},
			},
			CN:         hostname,
			Hosts:      []string{hostname, "www." + hostname},
			KeyRequest: &param,
		}
		certBytes, _, keyBytes, err := New(req)
		if err != nil {
			t.Fatal("InitCA failed:", err)
		}
		key, err := helpers.ParsePrivateKeyPEM(keyBytes)
		if err != nil {
			t.Fatal("InitCA private key parsing failed:", err)
		}
		cert, err := helpers.ParseCertificatePEM(certBytes)
		if err != nil {
			t.Fatal("InitCA cert parsing failed:", err)
		}

		// Verify key parameters.
		switch req.KeyRequest.Algo() {
		case "rsa":
			if cert.PublicKey.(*rsa.PublicKey).N.BitLen() != param.Size() {
				t.Fatal("Cert key length mismatch.")
			}
			if key.(*rsa.PrivateKey).N.BitLen() != param.Size() {
				t.Fatal("Private key length mismatch.")
			}
		case "ecdsa":
			if cert.PublicKey.(*ecdsa.PublicKey).Curve.Params().BitSize != param.Size() {
				t.Fatal("Cert key length mismatch.")
			}
			if key.(*ecdsa.PrivateKey).Curve.Params().BitSize != param.Size() {
				t.Fatal("Private key length mismatch.")
			}
		}

		// Start a signer
		var CAPolicy = &config.Signing{
			Default: &config.SigningProfile{
				Usage:        []string{"cert sign", "crl sign"},
				ExpiryString: "300s",
				Expiry:       300 * time.Second,
				CA:           true,
			},
		}
		s, err := local.NewSigner(key, cert, signer.DefaultSigAlgo(key), nil)
		if err != nil {
			t.Fatal("Signer Creation error:", err)
		}
		s.SetPolicy(CAPolicy)

		// Sign RSA and ECDSA customer CSRs.
		for _, csrFile := range csrFiles {
			csrBytes, err := ioutil.ReadFile(csrFile)
			if err != nil {
				t.Fatal("CSR loading error:", err)
			}
			req := signer.SignRequest{
				Request: string(csrBytes),
				Hosts:   signer.SplitHosts(hostname),
				Profile: "",
				Label:   "",
			}

			bytes, err := s.Sign(req)
			if err != nil {
				t.Fatal(err)
			}
			customerCert, _ := helpers.ParseCertificatePEM(bytes)
			if customerCert.SignatureAlgorithm != s.SigAlgo() {
				t.Fatal("Signature Algorithm mismatch")
			}
			err = customerCert.CheckSignatureFrom(cert)
			if err != nil {
				t.Fatal("Signing CSR failed.", err)
			}
		}

	}
}

func TestInvalidCryptoParams(t *testing.T) {
	var req *csr.CertificateRequest
	hostname := "cloudflare.com"
	for _, invalidParam := range invalidCryptoParams {
		req = &csr.CertificateRequest{
			Names: []csr.Name{
				{
					C:  "US",
					ST: "California",
					L:  "San Francisco",
					O:  "CloudFlare",
					OU: "Systems Engineering",
				},
			},
			CN:         hostname,
			Hosts:      []string{hostname, "www." + hostname},
			KeyRequest: &invalidParam,
		}
		_, _, _, err := New(req)
		if err == nil {
			t.Fatal("InitCA with bad params should fail:", err)
		}

		if !strings.Contains(err.Error(), `"code":2400`) {
			t.Fatal(err)
		}
	}
}

type validation struct {
	r *csr.CertificateRequest
	v bool
}

var testValidations = []validation{
	{&csr.CertificateRequest{}, false},
	{&csr.CertificateRequest{
		CN: "test CA",
	}, true},
	{&csr.CertificateRequest{
		Names: []csr.Name{{}},
	}, false},
	{&csr.CertificateRequest{
		Names: []csr.Name{
			{O: "Example CA"},
		},
	}, true},
}

func TestValidations(t *testing.T) {
	for i, tv := range testValidations {
		err := validator(tv.r)
		if tv.v && err != nil {
			t.Fatalf("%v", err)
		}

		if !tv.v && err == nil {
			t.Fatalf("%d: expected error, but no error was reported", i)
		}
	}
}

func TestRenewRSA(t *testing.T) {
	certPEM, err := RenewFromPEM(testRSACAFile, testRSACAKeyFile)
	if err != nil {
		t.Fatal(err)
	}

	// must parse ok
	cert, err := helpers.ParseCertificatePEM(certPEM)
	if err != nil {
		t.Fatal(err)
	}

	if !cert.IsCA {
		t.Fatal("renewed CA certificate is not CA")
	}

	// cert expiry must be 5 minutes
	expiry := cert.NotAfter.Sub(cert.NotBefore).Seconds()
	if expiry >= 301 || expiry <= 299 {
		t.Fatal("expiry is not correct")
	}

	// check subject

	if cert.Subject.CommonName != "" {
		t.Fatal("Bad CommonName")
	}

	if len(cert.Subject.Country) != 1 || cert.Subject.Country[0] != "US" {
		t.Fatal("Bad Subject")
	}

	if len(cert.Subject.Organization) != 1 || cert.Subject.Organization[0] != "CloudFlare, Inc." {
		t.Fatal("Bad Subject")
	}
}

func TestRenewECDSA(t *testing.T) {
	certPEM, err := RenewFromPEM(testECDSACAFile, testECDSACAKeyFile)
	if err != nil {
		t.Fatal(err)
	}

	// must parse ok
	cert, err := helpers.ParseCertificatePEM(certPEM)
	if err != nil {
		t.Fatal(err)
	}

	if !cert.IsCA {
		t.Fatal("renewed CA certificate is not CA")
	}

	// cert expiry must be 5 minutes
	expiry := cert.NotAfter.Sub(cert.NotBefore).Seconds()
	if expiry >= 301 || expiry <= 299 {
		t.Fatal("expiry is not correct")
	}

	// check subject

	if cert.Subject.CommonName != "" {
		t.Fatal("Bad CommonName")
	}

	if len(cert.Subject.Country) != 1 || cert.Subject.Country[0] != "US" {
		t.Fatal("Bad Subject")
	}

	if len(cert.Subject.Organization) != 1 || cert.Subject.Organization[0] != "CloudFlare, Inc." {
		t.Fatal("Bad Subject")
	}
}

func TestRenewMismatch(t *testing.T) {
	_, err := RenewFromPEM(testECDSACAFile, testRSACAKeyFile)
	if err == nil {
		t.Fatal("Fail to detect cert/key mismatch")
	}
}
