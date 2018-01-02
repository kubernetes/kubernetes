package universal

import (
	"bytes"
	"io/ioutil"
	"math/big"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	apiinfo "github.com/cloudflare/cfssl/api/info"
	apisign "github.com/cloudflare/cfssl/api/signhandler"
	"github.com/cloudflare/cfssl/config"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/helpers/testsuite"
	"github.com/cloudflare/cfssl/info"
	"github.com/cloudflare/cfssl/signer"
)

const (
	testCaFile    = "../local/testdata/ca.pem"
	testCaKeyFile = "../local/testdata/ca_key.pem"
)

var expiry = 1 * time.Minute
var validLocalConfig = &config.Config{
	Signing: &config.Signing{
		Profiles: map[string]*config.SigningProfile{
			"valid": {
				Usage:  []string{"digital signature"},
				Expiry: expiry,
			},
		},
		Default: &config.SigningProfile{
			Usage:  []string{"digital signature"},
			Expiry: expiry,
		},
	},
}

var validMinimalRemoteConfig = `
{
  "signing": {
    "profiles": {
      "CA": {
        "usages": [
          "cert sign",
          "crl sign"
        ],
        "expiry": "720h",
        "auth_key": "ca-auth"
      }
    },
    "default": {
      "usages": [
        "digital signature",
        "email protection"
      ],
      "expiry": "8000h"
    }
  },
  "auth_keys": {
    "ca-auth": {
      "type": "standard",
      "key": "0123456789ABCDEF0123456789ABCDEF"
    }
  }
}`

var validMinimalUniversalConfig = `
{
  "signing": {
    "profiles": {
      "CA": {
        "usages": [
          "cert sign",
          "crl sign"
        ],
        "expiry": "720h",
        "auth_key": "local-auth",
        "auth_remote": {
          "remote": "localhost",
          "auth_key": "ca-auth"
        }
      },
      "email": {
        "usages": [
          "s/mime"
        ],
        "expiry": "720h"
      }
    },
    "default": {
      "usages": [
        "digital signature",
        "email protection"
      ],
      "expiry": "8000h"
    }
  },
  "auth_keys": {
    "local-auth": {
      "type": "standard",
      "key": "123456789ABCDEF0123456789ABCDEF0"
    },
    "ca-auth": {
      "type": "standard",
      "key": "0123456789ABCDEF0123456789ABCDEF"
    }
  },
  "remotes": {
    "localhost": "127.0.0.1:1234"
  }
}`

var validRemoteConfig = `
{
  "signing": {
    "profiles": {
      "CA": {
        "usages": [
          "cert sign",
          "crl sign"
        ],
        "expiry": "720h",
        "auth_key": "ca-auth"
      },
      "ipsec": {
        "usages": [
          "ipsec tunnel"
        ],
        "expiry": "720h"
      }
    },
    "default": {
      "usages": [
        "digital signature",
        "email protection"
      ],
      "expiry": "8000h"
    }
  },
  "auth_keys": {
    "ca-auth": {
      "type": "standard",
      "key": "0123456789ABCDEF0123456789ABCDEF"
    }
  }
}`

var validUniversalConfig = `
{
  "signing": {
    "profiles": {
      "CA": {
        "usages": [
          "cert sign",
          "crl sign"
        ],
        "expiry": "720h",
        "auth_key": "local-auth",
        "auth_remote": {
          "remote": "localhost",
          "auth_key": "ca-auth"
        }
      },
      "ipsec": {
        "usages": [
          "ipsec tunnel"
        ],
        "expiry": "720h",
		"remote": "localhost"
      },
      "email": {
        "usages": [
          "s/mime"
        ],
        "expiry": "720h"
      }
    },
    "default": {
      "usages": [
        "digital signature",
        "email protection"
      ],
      "expiry": "8000h"
    }
  },
  "auth_keys": {
    "local-auth": {
      "type": "standard",
      "key": "123456789ABCDEF0123456789ABCDEF0"
    },
    "ca-auth": {
      "type": "standard",
      "key": "0123456789ABCDEF0123456789ABCDEF"
    }
  },
  "remotes": {
    "localhost": "127.0.0.1:1234"
  }
}`

var validNoAuthRemoteConfig = `
{
  "signing": {
    "profiles": {
      "CA": {
        "usages": [
          "cert sign",
          "crl sign"
        ],
        "expiry": "720h"
      },
      "ipsec": {
        "usages": [
          "ipsec tunnel"
        ],
        "expiry": "720h"
      }
    },
    "default": {
      "usages": [
        "digital signature",
        "email protection"
      ],
      "expiry": "8000h"
    }
  }
}`

var validNoAuthUniversalConfig = `
{
  "signing": {
    "profiles": {
      "CA": {
        "usages": [
          "cert sign",
          "crl sign"
        ],
        "expiry": "720h",
		"remote": "localhost"
      },
      "ipsec": {
        "usages": [
          "ipsec tunnel"
        ],
        "expiry": "720h",
		"remote": "localhost"
      },
      "email": {
        "usages": [
          "s/mime"
        ],
        "expiry": "720h"
      }
    },
    "default": {
      "usages": [
        "digital signature",
        "email protection"
      ],
      "expiry": "8000h"
    }
  },
  "remotes": {
    "localhost": "127.0.0.1:1234"
  }
}`

func TestNewSigner(t *testing.T) {
	h := map[string]string{
		"key-file":  testCaKeyFile,
		"cert-file": testCaFile,
	}

	r := &Root{
		Config:      h,
		ForceRemote: false,
	}

	_, err := NewSigner(*r, validLocalConfig.Signing)
	if err != nil {
		t.Fatal(err)
	}
}

func checkInfo(t *testing.T, s signer.Signer, name string, profile *config.SigningProfile) {
	req := info.Req{
		Profile: name,
	}
	resp, err := s.Info(req)
	if err != nil {
		t.Fatal("remote info failed:", err)
	}

	if strings.Join(profile.Usage, ",") != strings.Join(resp.Usage, ",") {
		t.Fatalf("Expected usage for profile %s to be %+v, got %+v", name, profile.Usage, resp.Usage)
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

func TestUniversalRemoteAndLocalInfo(t *testing.T) {
	// set up remote server
	remoteConfig := testsuite.NewConfig(t, []byte(validMinimalRemoteConfig))
	remoteServer := newTestInfoServer(t, newTestUniversalSigner(t, remoteConfig.Signing))
	defer closeTestServer(t, remoteServer)

	universalConfig := testsuite.NewConfig(t, []byte(validMinimalUniversalConfig))
	// override with test server address, ignore url prefix "http://"
	for name, profile := range universalConfig.Signing.Profiles {
		if profile.RemoteServer != "" {
			universalConfig.Signing.Profiles[name].RemoteServer = remoteServer.URL[7:]
		}
	}
	s := newTestUniversalSigner(t, universalConfig.Signing)

	for name, profile := range universalConfig.Signing.Profiles {
		checkInfo(t, s, name, profile)
	}

	// add check for default profile
	checkInfo(t, s, "", universalConfig.Signing.Default)
}

func TestUniversalMultipleRemoteAndLocalInfo(t *testing.T) {
	// set up remote server
	remoteConfig := testsuite.NewConfig(t, []byte(validRemoteConfig))
	remoteServer := newTestInfoServer(t, newTestUniversalSigner(t, remoteConfig.Signing))
	defer closeTestServer(t, remoteServer)

	universalConfig := testsuite.NewConfig(t, []byte(validUniversalConfig))
	// override with test server address, ignore url prefix "http://"
	for name, profile := range universalConfig.Signing.Profiles {
		if profile.RemoteServer != "" {
			universalConfig.Signing.Profiles[name].RemoteServer = remoteServer.URL[7:]
		}
	}
	s := newTestUniversalSigner(t, universalConfig.Signing)

	for name, profile := range universalConfig.Signing.Profiles {
		checkInfo(t, s, name, profile)
	}

	// add check for default profile
	checkInfo(t, s, "", universalConfig.Signing.Default)
}

func TestUniversalRemoteAndLocalSign(t *testing.T) {
	// set up remote server
	remoteConfig := testsuite.NewConfig(t, []byte(validNoAuthRemoteConfig))
	remoteServer := newTestSignServer(t, newTestUniversalSigner(t, remoteConfig.Signing))
	defer closeTestServer(t, remoteServer)

	universalConfig := testsuite.NewConfig(t, []byte(validNoAuthUniversalConfig))
	// override with test server address, ignore url prefix "http://"
	for name, profile := range universalConfig.Signing.Profiles {
		if profile.RemoteServer != "" {
			universalConfig.Signing.Profiles[name].RemoteServer = remoteServer.URL[7:]
		}
	}
	s := newTestUniversalSigner(t, universalConfig.Signing)

	checkSign := func(name string, profile *config.SigningProfile) {
		hosts := []string{"cloudflare.com"}
		for _, test := range testsuite.CSRTests {
			csr, err := ioutil.ReadFile(test.File)
			if err != nil {
				t.Fatalf("CSR loading error (%s): %v", name, err)
			}
			testSerial := big.NewInt(0x7007F)

			certBytes, err := s.Sign(signer.SignRequest{
				Hosts:   hosts,
				Request: string(csr),
				Serial:  testSerial,
				Profile: name,
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
				ku, _, _ := profile.Usages()
				if cert.KeyUsage != ku {
					t.Fatalf("Key usage was incorrect expected %+v, got %+v", ku, cert.KeyUsage)
				}
			}
		}
	}

	for name, profile := range universalConfig.Signing.Profiles {
		checkSign(name, profile)
	}

	// add check for default profile
	checkSign("", universalConfig.Signing.Default)
}

func newTestUniversalSigner(t *testing.T, policy *config.Signing) signer.Signer {
	h := map[string]string{
		"key-file":  testCaKeyFile,
		"cert-file": testCaFile,
	}

	r := &Root{
		Config:      h,
		ForceRemote: false,
	}

	s, err := NewSigner(*r, policy)
	if err != nil {
		t.Fatal("fail to init universal signer:", err)
	}

	return s
}

func newTestSignHandler(t *testing.T, s signer.Signer) (h http.Handler) {
	h, err := apisign.NewHandlerFromSigner(s)
	if err != nil {
		t.Fatal(err)
	}
	return
}

func newTestInfoHandler(t *testing.T, s signer.Signer) (h http.Handler) {
	h, err := apiinfo.NewHandler(s)
	if err != nil {
		t.Fatal(err)
	}
	return
}

func newTestSignServer(t *testing.T, s signer.Signer) *httptest.Server {
	mux := http.NewServeMux()
	mux.Handle("/api/v1/cfssl/sign", newTestSignHandler(t, s))
	ts := httptest.NewUnstartedServer(mux)
	ts.Start()
	t.Log(ts.URL)
	return ts
}

func newTestInfoServer(t *testing.T, s signer.Signer) *httptest.Server {
	mux := http.NewServeMux()
	mux.Handle("/api/v1/cfssl/info", newTestInfoHandler(t, s))
	ts := httptest.NewUnstartedServer(mux)
	ts.Start()
	t.Log(ts.URL)
	return ts
}

func closeTestServer(t *testing.T, ts *httptest.Server) {
	t.Log("Finalizing test server.")
	ts.Close()
}
