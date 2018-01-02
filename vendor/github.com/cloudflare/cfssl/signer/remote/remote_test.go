package remote

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"math/big"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	apiinfo "github.com/cloudflare/cfssl/api/info"
	apisign "github.com/cloudflare/cfssl/api/signhandler"
	"github.com/cloudflare/cfssl/config"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/helpers/testsuite"
	"github.com/cloudflare/cfssl/info"
	"github.com/cloudflare/cfssl/signer"
	"github.com/cloudflare/cfssl/signer/local"
)

const (
	testCaFile    = "testdata/ca.pem"
	testCaKeyFile = "testdata/ca_key.pem"
)

var validMinimalRemoteConfig = `
{
	"signing": {
		"default": {
			"remote": "localhost"
		}
	},
	"remotes": {
		"localhost": "127.0.0.1:80"
	}
}`

var validMinimalAuthRemoteConfig = `
{
	"signing": {
		"default": {
			"auth_key": "sample",
			"remote": "localhost"
		}
	},
	"auth_keys": {
		"sample": {
			"type":"standard",
			"key":"0123456789ABCDEF0123456789ABCDEF"
		}
	},
	"remotes": {
		"localhost": "127.0.0.1:80"
	}
}`

func TestNewSigner(t *testing.T) {
	remoteConfig := testsuite.NewConfig(t, []byte(validMinimalRemoteConfig))

	_, err := NewSigner(remoteConfig.Signing)
	if err != nil {
		t.Fatal("fail to init remote signer:", err)
	}
}

func TestNewAuthSigner(t *testing.T) {
	remoteAuthConfig := testsuite.NewConfig(t, []byte(validMinimalAuthRemoteConfig))

	_, err := NewSigner(remoteAuthConfig.Signing)
	if err != nil {
		t.Fatal("fail to init remote signer:", err)
	}
}

func TestRemoteInfo(t *testing.T) {
	remoteServer := newTestInfoServer(t)
	defer closeTestServer(t, remoteServer)

	remoteConfig := testsuite.NewConfig(t, []byte(validMinimalRemoteConfig))
	// override with test server address, ignore url prefix "http://"
	remoteConfig.Signing.OverrideRemotes(remoteServer.URL[7:])
	s := newRemoteSigner(t, remoteConfig.Signing)
	req := info.Req{}
	resp, err := s.Info(req)
	if err != nil {
		t.Fatal("remote info failed:", err)
	}

	caBytes, err := ioutil.ReadFile(testCaFile)
	caBytes = bytes.TrimSpace(caBytes)
	if err != nil {
		t.Fatal("fail to read test CA cert:", err)
	}

	if bytes.Compare(caBytes, []byte(resp.Certificate)) != 0 {
		t.Fatal("Get a different CA cert through info api.", len(resp.Certificate), len(caBytes))
	}
}

func TestRemoteSign(t *testing.T) {
	remoteServer := newTestSignServer(t)
	defer closeTestServer(t, remoteServer)

	remoteConfig := testsuite.NewConfig(t, []byte(validMinimalRemoteConfig))
	// override with test server address, ignore url prefix "http://"
	remoteConfig.Signing.OverrideRemotes(remoteServer.URL[7:])
	s := newRemoteSigner(t, remoteConfig.Signing)

	hosts := []string{"cloudflare.com"}
	for _, test := range testsuite.CSRTests {
		csr, err := ioutil.ReadFile(test.File)
		if err != nil {
			t.Fatal("CSR loading error:", err)
		}
		testSerial := big.NewInt(0x7007F)
		certBytes, err := s.Sign(signer.SignRequest{
			Hosts:   hosts,
			Request: string(csr),
			Serial:  testSerial,
		})
		if test.ErrorCallback != nil {
			test.ErrorCallback(t, err)
		} else {
			if err != nil {
				t.Fatalf("Expected no error. Got %s. Param %s %d", err.Error(), test.KeyAlgo, test.KeyLen)
			}
			cert, err := helpers.ParseCertificatePEM(certBytes)
			if err != nil {
				t.Fatal("Fail to parse returned certificate:", err)
			}
			sn := fmt.Sprintf("%X", cert.SerialNumber)
			if sn != "7007F" {
				t.Fatal("Serial Number was incorrect:", sn)
			}
		}
	}
}

func TestRemoteSignBadServerAndOverride(t *testing.T) {
	remoteServer := newTestSignServer(t)
	defer closeTestServer(t, remoteServer)

	// remoteConfig contains port 80 that no test server will listen on
	remoteConfig := testsuite.NewConfig(t, []byte(validMinimalRemoteConfig))
	s := newRemoteSigner(t, remoteConfig.Signing)

	hosts := []string{"cloudflare.com"}
	csr, err := ioutil.ReadFile("../local/testdata/rsa2048.csr")
	if err != nil {
		t.Fatal("CSR loading error:", err)
	}

	_, err = s.Sign(signer.SignRequest{Hosts: hosts, Request: string(csr)})
	if err == nil {
		t.Fatal("Should return error")
	}

	remoteConfig.Signing.OverrideRemotes(remoteServer.URL[7:])
	s.SetPolicy(remoteConfig.Signing)
	certBytes, err := s.Sign(signer.SignRequest{
		Hosts:   hosts,
		Request: string(csr),
		Serial:  big.NewInt(1),
	})
	if err != nil {
		t.Fatalf("Expected no error. Got %s.", err.Error())
	}
	_, err = helpers.ParseCertificatePEM(certBytes)
	if err != nil {
		t.Fatal("Fail to parse returned certificate:", err)
	}

}

// helper functions
func newRemoteSigner(t *testing.T, policy *config.Signing) *Signer {
	s, err := NewSigner(policy)
	if err != nil {
		t.Fatal("fail to init remote signer:", err)
	}

	return s
}

func newTestSignHandler(t *testing.T) (h http.Handler) {
	h, err := newHandler(t, testCaFile, testCaKeyFile, "sign")
	if err != nil {
		t.Fatal(err)
	}
	return
}

func newTestInfoHandler(t *testing.T) (h http.Handler) {
	h, err := newHandler(t, testCaFile, testCaKeyFile, "info")
	if err != nil {
		t.Fatal(err)
	}
	return
}

func newTestSignServer(t *testing.T) *httptest.Server {
	mux := http.NewServeMux()
	mux.Handle("/api/v1/cfssl/sign", newTestSignHandler(t))
	ts := httptest.NewUnstartedServer(mux)
	ts.Start()
	t.Log(ts.URL)
	return ts
}

func newTestInfoServer(t *testing.T) *httptest.Server {
	mux := http.NewServeMux()
	mux.Handle("/api/v1/cfssl/info", newTestInfoHandler(t))
	ts := httptest.NewUnstartedServer(mux)
	ts.Start()
	t.Log(ts.URL)
	return ts
}

func closeTestServer(t *testing.T, ts *httptest.Server) {
	t.Log("Finalizing test server.")
	ts.Close()
}

// newHandler generates a new sign handler (or info handler) using the certificate
// authority private key and certficate to sign certificates.
func newHandler(t *testing.T, caFile, caKeyFile, op string) (http.Handler, error) {
	var expiry = 1 * time.Minute
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

				ClientProvidesSerialNumbers: true,
			},
		},
	}
	s, err := local.NewSignerFromFile(testCaFile, testCaKeyFile, CAConfig.Signing)
	if err != nil {
		t.Fatal(err)
	}
	if op == "sign" {
		return apisign.NewHandlerFromSigner(s)
	} else if op == "info" {
		return apiinfo.NewHandler(s)
	}

	t.Fatal("Bad op code")
	return nil, nil
}
