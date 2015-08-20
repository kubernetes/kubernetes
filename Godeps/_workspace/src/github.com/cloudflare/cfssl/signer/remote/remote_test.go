package remote

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/cloudflare/cfssl/api"
	apiinfo "github.com/cloudflare/cfssl/api/info"
	"github.com/cloudflare/cfssl/config"
	"github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/info"
	"github.com/cloudflare/cfssl/log"
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
	remoteConfig := newConfig(t, []byte(validMinimalRemoteConfig))

	_, err := NewSigner(remoteConfig.Signing)
	if err != nil {
		t.Fatal("fail to init remote signer:", err)
	}
}

func TestNewAuthSigner(t *testing.T) {
	remoteAuthConfig := newConfig(t, []byte(validMinimalAuthRemoteConfig))

	_, err := NewSigner(remoteAuthConfig.Signing)
	if err != nil {
		t.Fatal("fail to init remote signer:", err)
	}
}

func TestRemoteInfo(t *testing.T) {
	remoteServer := newTestInfoServer(t)
	defer closeTestServer(t, remoteServer)

	remoteConfig := newConfig(t, []byte(validMinimalRemoteConfig))
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

type csrTest struct {
	file    string
	keyAlgo string
	keyLen  int
	// Error checking function
	errorCallback func(*testing.T, error)
}

var csrTests = []csrTest{
	{
		file:          "../local/testdata/rsa2048.csr",
		keyAlgo:       "rsa",
		keyLen:        2048,
		errorCallback: nil,
	},
	{
		file:          "../local/testdata/rsa3072.csr",
		keyAlgo:       "rsa",
		keyLen:        3072,
		errorCallback: nil,
	},
	{
		file:          "../local/testdata/rsa4096.csr",
		keyAlgo:       "rsa",
		keyLen:        4096,
		errorCallback: nil,
	},
	{
		file:          "../local/testdata/ecdsa256.csr",
		keyAlgo:       "ecdsa",
		keyLen:        256,
		errorCallback: nil,
	},
	{
		file:          "../local/testdata/ecdsa384.csr",
		keyAlgo:       "ecdsa",
		keyLen:        384,
		errorCallback: nil,
	},
	{
		file:          "../local/testdata/ecdsa521.csr",
		keyAlgo:       "ecdsa",
		keyLen:        521,
		errorCallback: nil,
	},
}

func TestRemoteSign(t *testing.T) {
	remoteServer := newTestSignServer(t)
	defer closeTestServer(t, remoteServer)

	remoteConfig := newConfig(t, []byte(validMinimalRemoteConfig))
	// override with test server address, ignore url prefix "http://"
	remoteConfig.Signing.OverrideRemotes(remoteServer.URL[7:])
	s := newRemoteSigner(t, remoteConfig.Signing)

	hosts := []string{"cloudflare.com"}
	for _, test := range csrTests {
		csr, err := ioutil.ReadFile(test.file)
		if err != nil {
			t.Fatal("CSR loading error:", err)
		}
		testSeq := "7007F"
		certBytes, err := s.Sign(signer.SignRequest{Hosts: hosts, Request: string(csr), SerialSeq: testSeq})
		if test.errorCallback != nil {
			test.errorCallback(t, err)
		} else {
			if err != nil {
				t.Fatalf("Expected no error. Got %s. Param %s %d", err.Error(), test.keyAlgo, test.keyLen)
			}
			cert, err := helpers.ParseCertificatePEM(certBytes)
			if err != nil {
				t.Fatal("Fail to parse returned certificate:", err)
			}
			sn := fmt.Sprintf("%X", cert.SerialNumber)
			if sn[0:len(testSeq)] != testSeq {
				t.Fatal("Serial Number did not begin with seq:", sn)
			}
			if len(sn) != len(testSeq)+16 {
				t.Fatal("Serial Number unexpected length:", sn)
			}
		}
	}
}

func TestRemoteSignBadServerAndOverride(t *testing.T) {
	remoteServer := newTestSignServer(t)
	defer closeTestServer(t, remoteServer)

	// remoteConfig contains port 80 that no test server will listen on
	remoteConfig := newConfig(t, []byte(validMinimalRemoteConfig))
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
	certBytes, err := s.Sign(signer.SignRequest{Hosts: hosts, Request: string(csr)})
	if err != nil {
		t.Fatalf("Expected no error. Got %s.", err.Error())
	}
	_, err = helpers.ParseCertificatePEM(certBytes)
	if err != nil {
		t.Fatal("Fail to parse returned certificate:", err)
	}

}

// helper functions
func newConfig(t *testing.T, configBytes []byte) *config.Config {
	conf, err := config.LoadConfig([]byte(configBytes))
	if err != nil {
		t.Fatal("config loading error:", err)
	}
	if !conf.Valid() {
		t.Fatal("config is not valid")
	}
	return conf
}

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
				"signature": &config.SigningProfile{
					Usage:  []string{"digital signature"},
					Expiry: expiry,
				},
			},
			Default: &config.SigningProfile{
				Usage:        []string{"cert sign", "crl sign"},
				ExpiryString: "43800h",
				Expiry:       expiry,
				CA:           true,
				UseSerialSeq: true,
			},
		},
	}
	s, err := local.NewSignerFromFile(testCaFile, testCaKeyFile, CAConfig.Signing)
	if err != nil {
		t.Fatal(err)
	}
	if op == "sign" {
		return NewSignHandlerFromSigner(s)
	} else if op == "info" {
		return apiinfo.NewHandler(s)
	}

	t.Fatal("Bad op code")
	return nil, nil
}

// NewSignHandlerFromSigner generates a new SignHandler directly from
// an existing signer.
func NewSignHandlerFromSigner(s signer.Signer) (h http.Handler, err error) {
	policy := s.Policy()
	if policy == nil {
		err = errors.New(errors.PolicyError, errors.InvalidPolicy)
		return
	}

	// Sign will only respond for profiles that have no auth provider.
	// So if all of the profiles require authentication, we return an error.
	haveUnauth := (policy.Default.Provider == nil)
	for _, profile := range policy.Profiles {
		if !haveUnauth {
			break
		}
		haveUnauth = (profile.Provider == nil)
	}

	if !haveUnauth {
		err = errors.New(errors.PolicyError, errors.InvalidPolicy)
		return
	}

	return &api.HTTPHandler{
		Handler: &SignHandler{
			signer: s,
		},
		Methods: []string{"POST"},
	}, nil
}

// A SignHandler accepts requests with a hostname and certficate
// parameter (which should be PEM-encoded) and returns a new signed
// certificate. It includes upstream servers indexed by their
// profile name.
type SignHandler struct {
	signer signer.Signer
}

// Handle responds to requests for the CA to sign the certificate request
// present in the "certificate_request" parameter for the host named
// in the "hostname" parameter. The certificate should be PEM-encoded. If
// provided, subject information from the "subject" parameter will be used
// in place of the subject information from the CSR.
func (h *SignHandler) Handle(w http.ResponseWriter, r *http.Request) error {
	log.Info("signature request received")

	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		return err
	}
	r.Body.Close()

	var req signer.SignRequest
	err = json.Unmarshal(body, &req)
	if err != nil {
		return err
	}

	if len(req.Hosts) == 0 {
		return errors.NewBadRequestString("missing paratmeter 'hosts'")
	}

	if req.Request == "" {
		return errors.NewBadRequestString("missing parameter 'certificate_request'")
	}

	var cert []byte
	var profile *config.SigningProfile

	policy := h.signer.Policy()
	if policy != nil && policy.Profiles != nil && req.Profile != "" {
		profile = policy.Profiles[req.Profile]
	}

	if profile == nil && policy != nil {
		profile = policy.Default
	}

	if profile.Provider != nil {
		log.Error("profile requires authentication")
		return errors.NewBadRequestString("authentication required")
	}

	cert, err = h.signer.Sign(req)
	if err != nil {
		log.Warningf("failed to sign request: %v", err)
		return err
	}

	result := map[string]string{"certificate": string(cert)}
	log.Info("wrote response")
	return api.SendResponse(w, result)
}
