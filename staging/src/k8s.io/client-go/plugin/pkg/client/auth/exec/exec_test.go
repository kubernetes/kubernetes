/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package exec

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io/ioutil"
	"math/big"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/pkg/apis/clientauthentication"
	"k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/client-go/transport"
)

var (
	certData = []byte(`-----BEGIN CERTIFICATE-----
MIIC6jCCAdSgAwIBAgIBCzALBgkqhkiG9w0BAQswIzEhMB8GA1UEAwwYMTAuMTMu
MTI5LjEwNkAxNDIxMzU5MDU4MB4XDTE1MDExNTIyMDEzMVoXDTE2MDExNTIyMDEz
MlowGzEZMBcGA1UEAxMQb3BlbnNoaWZ0LWNsaWVudDCCASIwDQYJKoZIhvcNAQEB
BQADggEPADCCAQoCggEBAKtdhz0+uCLXw5cSYns9rU/XifFSpb/x24WDdrm72S/v
b9BPYsAStiP148buylr1SOuNi8sTAZmlVDDIpIVwMLff+o2rKYDicn9fjbrTxTOj
lI4pHJBH+JU3AJ0tbajupioh70jwFS0oYpwtneg2zcnE2Z4l6mhrj2okrc5Q1/X2
I2HChtIU4JYTisObtin10QKJX01CLfYXJLa8upWzKZ4/GOcHG+eAV3jXWoXidtjb
1Usw70amoTZ6mIVCkiu1QwCoa8+ycojGfZhvqMsAp1536ZcCul+Na+AbCv4zKS7F
kQQaImVrXdUiFansIoofGlw/JNuoKK6ssVpS5Ic3pgcCAwEAAaM1MDMwDgYDVR0P
AQH/BAQDAgCgMBMGA1UdJQQMMAoGCCsGAQUFBwMCMAwGA1UdEwEB/wQCMAAwCwYJ
KoZIhvcNAQELA4IBAQCKLREH7bXtXtZ+8vI6cjD7W3QikiArGqbl36bAhhWsJLp/
p/ndKz39iFNaiZ3GlwIURWOOKx3y3GA0x9m8FR+Llthf0EQ8sUjnwaknWs0Y6DQ3
jjPFZOpV3KPCFrdMJ3++E3MgwFC/Ih/N2ebFX9EcV9Vcc6oVWMdwT0fsrhu683rq
6GSR/3iVX1G/pmOiuaR0fNUaCyCfYrnI4zHBDgSfnlm3vIvN2lrsR/DQBakNL8DJ
HBgKxMGeUPoneBv+c8DMXIL0EhaFXRlBv9QW45/GiAIOuyFJ0i6hCtGZpJjq4OpQ
BRjCI+izPzFTjsxD4aORE+WOkyWFCGPWKfNejfw0
-----END CERTIFICATE-----`)
	keyData = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEAq12HPT64ItfDlxJiez2tT9eJ8VKlv/HbhYN2ubvZL+9v0E9i
wBK2I/Xjxu7KWvVI642LyxMBmaVUMMikhXAwt9/6jaspgOJyf1+NutPFM6OUjikc
kEf4lTcAnS1tqO6mKiHvSPAVLShinC2d6DbNycTZniXqaGuPaiStzlDX9fYjYcKG
0hTglhOKw5u2KfXRAolfTUIt9hcktry6lbMpnj8Y5wcb54BXeNdaheJ22NvVSzDv
RqahNnqYhUKSK7VDAKhrz7JyiMZ9mG+oywCnXnfplwK6X41r4BsK/jMpLsWRBBoi
ZWtd1SIVqewiih8aXD8k26gorqyxWlLkhzemBwIDAQABAoIBAD2XYRs3JrGHQUpU
FkdbVKZkvrSY0vAZOqBTLuH0zUv4UATb8487anGkWBjRDLQCgxH+jucPTrztekQK
aW94clo0S3aNtV4YhbSYIHWs1a0It0UdK6ID7CmdWkAj6s0T8W8lQT7C46mWYVLm
5mFnCTHi6aB42jZrqmEpC7sivWwuU0xqj3Ml8kkxQCGmyc9JjmCB4OrFFC8NNt6M
ObvQkUI6Z3nO4phTbpxkE1/9dT0MmPIF7GhHVzJMS+EyyRYUDllZ0wvVSOM3qZT0
JMUaBerkNwm9foKJ1+dv2nMKZZbJajv7suUDCfU44mVeaEO+4kmTKSGCGjjTBGkr
7L1ySDECgYEA5ElIMhpdBzIivCuBIH8LlUeuzd93pqssO1G2Xg0jHtfM4tz7fyeI
cr90dc8gpli24dkSxzLeg3Tn3wIj/Bu64m2TpZPZEIlukYvgdgArmRIPQVxerYey
OkrfTNkxU1HXsYjLCdGcGXs5lmb+K/kuTcFxaMOs7jZi7La+jEONwf8CgYEAwCs/
rUOOA0klDsWWisbivOiNPII79c9McZCNBqncCBfMUoiGe8uWDEO4TFHN60vFuVk9
8PkwpCfvaBUX+ajvbafIfHxsnfk1M04WLGCeqQ/ym5Q4sQoQOcC1b1y9qc/xEWfg
nIUuia0ukYRpl7qQa3tNg+BNFyjypW8zukUAC/kCgYB1/Kojuxx5q5/oQVPrx73k
2bevD+B3c+DYh9MJqSCNwFtUpYIWpggPxoQan4LwdsmO0PKzocb/ilyNFj4i/vII
NToqSc/WjDFpaDIKyuu9oWfhECye45NqLWhb/6VOuu4QA/Nsj7luMhIBehnEAHW+
GkzTKM8oD1PxpEG3nPKXYQKBgQC6AuMPRt3XBl1NkCrpSBy/uObFlFaP2Enpf39S
3OZ0Gv0XQrnSaL1kP8TMcz68rMrGX8DaWYsgytstR4W+jyy7WvZwsUu+GjTJ5aMG
77uEcEBpIi9CBzivfn7hPccE8ZgqPf+n4i6q66yxBJflW5xhvafJqDtW2LcPNbW/
bvzdmQKBgExALRUXpq+5dbmkdXBHtvXdRDZ6rVmrnjy4nI5bPw+1GqQqk6uAR6B/
F6NmLCQOO4PDG/cuatNHIr2FrwTmGdEL6ObLUGWn9Oer9gJhHVqqsY5I4sEPo4XX
stR0Yiw0buV6DL/moUO0HIM9Bjh96HJp+LxiIS6UCdIhMPp5HoQa
-----END RSA PRIVATE KEY-----`)
	validCert *tls.Certificate
)

func init() {
	cert, err := tls.X509KeyPair(certData, keyData)
	if err != nil {
		panic(err)
	}
	validCert = &cert
}

func TestCacheKey(t *testing.T) {
	c1 := &api.ExecConfig{
		Command: "foo-bar",
		Args:    []string{"1", "2"},
		Env: []api.ExecEnvVar{
			{Name: "3", Value: "4"},
			{Name: "5", Value: "6"},
			{Name: "7", Value: "8"},
		},
		APIVersion: "client.authentication.k8s.io/v1alpha1",
	}
	c2 := &api.ExecConfig{
		Command: "foo-bar",
		Args:    []string{"1", "2"},
		Env: []api.ExecEnvVar{
			{Name: "3", Value: "4"},
			{Name: "5", Value: "6"},
			{Name: "7", Value: "8"},
		},
		APIVersion: "client.authentication.k8s.io/v1alpha1",
	}
	c3 := &api.ExecConfig{
		Command: "foo-bar",
		Args:    []string{"1", "2"},
		Env: []api.ExecEnvVar{
			{Name: "3", Value: "4"},
			{Name: "5", Value: "6"},
		},
		APIVersion: "client.authentication.k8s.io/v1alpha1",
	}
	key1 := cacheKey(c1)
	key2 := cacheKey(c2)
	key3 := cacheKey(c3)
	if key1 != key2 {
		t.Error("key1 and key2 didn't match")
	}
	if key1 == key3 {
		t.Error("key1 and key3 matched")
	}
	if key2 == key3 {
		t.Error("key2 and key3 matched")
	}
}

func compJSON(t *testing.T, got, want []byte) {
	t.Helper()
	gotJSON := &bytes.Buffer{}
	wantJSON := &bytes.Buffer{}

	if err := json.Indent(gotJSON, got, "", "  "); err != nil {
		t.Errorf("got invalid JSON: %v", err)
	}
	if err := json.Indent(wantJSON, want, "", "  "); err != nil {
		t.Errorf("want invalid JSON: %v", err)
	}
	g := strings.TrimSpace(gotJSON.String())
	w := strings.TrimSpace(wantJSON.String())
	if g != w {
		t.Errorf("wanted %q, got %q", w, g)
	}
}

func TestRefreshCreds(t *testing.T) {
	tests := []struct {
		name        string
		config      api.ExecConfig
		output      string
		interactive bool
		response    *clientauthentication.Response
		wantInput   string
		wantCreds   credentials
		wantExpiry  time.Time
		wantErr     bool
	}{
		{
			name: "basic-request",
			config: api.ExecConfig{
				APIVersion: "client.authentication.k8s.io/v1alpha1",
			},
			wantInput: `{
				"kind":"ExecCredential",
				"apiVersion":"client.authentication.k8s.io/v1alpha1",
				"spec": {}
			}`,
			output: `{
				"kind": "ExecCredential",
				"apiVersion": "client.authentication.k8s.io/v1alpha1",
				"status": {
					"token": "foo-bar"
				}
			}`,
			wantCreds: credentials{token: "foo-bar"},
		},
		{
			name: "interactive",
			config: api.ExecConfig{
				APIVersion: "client.authentication.k8s.io/v1alpha1",
			},
			interactive: true,
			wantInput: `{
				"kind":"ExecCredential",
				"apiVersion":"client.authentication.k8s.io/v1alpha1",
				"spec": {
					"interactive": true
				}
			}`,
			output: `{
				"kind": "ExecCredential",
				"apiVersion": "client.authentication.k8s.io/v1alpha1",
				"status": {
					"token": "foo-bar"
				}
			}`,
			wantCreds: credentials{token: "foo-bar"},
		},
		{
			name: "response",
			config: api.ExecConfig{
				APIVersion: "client.authentication.k8s.io/v1alpha1",
			},
			response: &clientauthentication.Response{
				Header: map[string][]string{
					"WWW-Authenticate": {`Basic realm="Access to the staging site", charset="UTF-8"`},
				},
				Code: 401,
			},
			wantInput: `{
				"kind":"ExecCredential",
				"apiVersion":"client.authentication.k8s.io/v1alpha1",
				"spec": {
					"response": {
						"header": {
							"WWW-Authenticate": [
								"Basic realm=\"Access to the staging site\", charset=\"UTF-8\""
							]
						},
						"code": 401
					}
				}
			}`,
			output: `{
				"kind": "ExecCredential",
				"apiVersion": "client.authentication.k8s.io/v1alpha1",
				"status": {
					"token": "foo-bar"
				}
			}`,
			wantCreds: credentials{token: "foo-bar"},
		},
		{
			name: "expiry",
			config: api.ExecConfig{
				APIVersion: "client.authentication.k8s.io/v1alpha1",
			},
			wantInput: `{
				"kind":"ExecCredential",
				"apiVersion":"client.authentication.k8s.io/v1alpha1",
				"spec": {}
			}`,
			output: `{
				"kind": "ExecCredential",
				"apiVersion": "client.authentication.k8s.io/v1alpha1",
				"status": {
					"token": "foo-bar",
					"expirationTimestamp": "2006-01-02T15:04:05Z"
				}
			}`,
			wantExpiry: time.Date(2006, 01, 02, 15, 04, 05, 0, time.UTC),
			wantCreds:  credentials{token: "foo-bar"},
		},
		{
			name: "no-group-version",
			config: api.ExecConfig{
				APIVersion: "client.authentication.k8s.io/v1alpha1",
			},
			wantInput: `{
				"kind":"ExecCredential",
				"apiVersion":"client.authentication.k8s.io/v1alpha1",
				"spec": {}
			}`,
			output: `{
				"kind": "ExecCredential",
				"status": {
					"token": "foo-bar"
				}
			}`,
			wantErr: true,
		},
		{
			name: "no-status",
			config: api.ExecConfig{
				APIVersion: "client.authentication.k8s.io/v1alpha1",
			},
			wantInput: `{
				"kind":"ExecCredential",
				"apiVersion":"client.authentication.k8s.io/v1alpha1",
				"spec": {}
			}`,
			output: `{
				"kind": "ExecCredential",
				"apiVersion":"client.authentication.k8s.io/v1alpha1"
			}`,
			wantErr: true,
		},
		{
			name: "no-creds",
			config: api.ExecConfig{
				APIVersion: "client.authentication.k8s.io/v1alpha1",
			},
			wantInput: `{
				"kind":"ExecCredential",
				"apiVersion":"client.authentication.k8s.io/v1alpha1",
				"spec": {}
			}`,
			output: `{
				"kind": "ExecCredential",
				"apiVersion":"client.authentication.k8s.io/v1alpha1",
				"status": {}
			}`,
			wantErr: true,
		},
		{
			name: "TLS credentials",
			config: api.ExecConfig{
				APIVersion: "client.authentication.k8s.io/v1alpha1",
			},
			wantInput: `{
				"kind":"ExecCredential",
				"apiVersion":"client.authentication.k8s.io/v1alpha1",
				"spec": {}
			}`,
			output: fmt.Sprintf(`{
				"kind": "ExecCredential",
				"apiVersion": "client.authentication.k8s.io/v1alpha1",
				"status": {
					"clientKeyData": %q,
					"clientCertificateData": %q
				}
			}`, keyData, certData),
			wantCreds: credentials{cert: validCert},
		},
		{
			name: "bad TLS credentials",
			config: api.ExecConfig{
				APIVersion: "client.authentication.k8s.io/v1alpha1",
			},
			wantInput: `{
				"kind":"ExecCredential",
				"apiVersion":"client.authentication.k8s.io/v1alpha1",
				"spec": {}
			}`,
			output: `{
				"kind": "ExecCredential",
				"apiVersion": "client.authentication.k8s.io/v1alpha1",
				"status": {
					"clientKeyData": "foo",
					"clientCertificateData": "bar"
				}
			}`,
			wantErr: true,
		},
		{
			name: "cert but no key",
			config: api.ExecConfig{
				APIVersion: "client.authentication.k8s.io/v1alpha1",
			},
			wantInput: `{
				"kind":"ExecCredential",
				"apiVersion":"client.authentication.k8s.io/v1alpha1",
				"spec": {}
			}`,
			output: fmt.Sprintf(`{
				"kind": "ExecCredential",
				"apiVersion": "client.authentication.k8s.io/v1alpha1",
				"status": {
					"clientCertificateData": %q
				}
			}`, certData),
			wantErr: true,
		},
		{
			name: "beta-basic-request",
			config: api.ExecConfig{
				APIVersion: "client.authentication.k8s.io/v1beta1",
			},
			output: `{
				"kind": "ExecCredential",
				"apiVersion": "client.authentication.k8s.io/v1beta1",
				"status": {
					"token": "foo-bar"
				}
			}`,
			wantCreds: credentials{token: "foo-bar"},
		},
		{
			name: "beta-expiry",
			config: api.ExecConfig{
				APIVersion: "client.authentication.k8s.io/v1beta1",
			},
			output: `{
				"kind": "ExecCredential",
				"apiVersion": "client.authentication.k8s.io/v1beta1",
				"status": {
					"token": "foo-bar",
					"expirationTimestamp": "2006-01-02T15:04:05Z"
				}
			}`,
			wantExpiry: time.Date(2006, 01, 02, 15, 04, 05, 0, time.UTC),
			wantCreds:  credentials{token: "foo-bar"},
		},
		{
			name: "beta-no-group-version",
			config: api.ExecConfig{
				APIVersion: "client.authentication.k8s.io/v1beta1",
			},
			output: `{
				"kind": "ExecCredential",
				"status": {
					"token": "foo-bar"
				}
			}`,
			wantErr: true,
		},
		{
			name: "beta-no-status",
			config: api.ExecConfig{
				APIVersion: "client.authentication.k8s.io/v1beta1",
			},
			output: `{
				"kind": "ExecCredential",
				"apiVersion":"client.authentication.k8s.io/v1beta1"
			}`,
			wantErr: true,
		},
		{
			name: "beta-no-token",
			config: api.ExecConfig{
				APIVersion: "client.authentication.k8s.io/v1beta1",
			},
			output: `{
				"kind": "ExecCredential",
				"apiVersion":"client.authentication.k8s.io/v1beta1",
				"status": {}
			}`,
			wantErr: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			c := test.config

			c.Command = "./testdata/test-plugin.sh"
			c.Env = append(c.Env, api.ExecEnvVar{
				Name:  "TEST_OUTPUT",
				Value: test.output,
			})

			a, err := newAuthenticator(newCache(), &c)
			if err != nil {
				t.Fatal(err)
			}

			stderr := &bytes.Buffer{}
			a.stderr = stderr
			a.interactive = test.interactive
			a.environ = func() []string { return nil }

			if err := a.refreshCredsLocked(test.response); err != nil {
				if !test.wantErr {
					t.Errorf("get token %v", err)
				}
				return
			}
			if test.wantErr {
				t.Fatal("expected error getting token")
			}

			if !reflect.DeepEqual(a.cachedCreds, &test.wantCreds) {
				t.Errorf("expected credentials %+v got %+v", &test.wantCreds, a.cachedCreds)
			}

			if !a.exp.Equal(test.wantExpiry) {
				t.Errorf("expected expiry %v got %v", test.wantExpiry, a.exp)
			}

			if test.wantInput == "" {
				if got := strings.TrimSpace(stderr.String()); got != "" {
					t.Errorf("expected no input parameters, got %q", got)
				}
				return
			}

			compJSON(t, stderr.Bytes(), []byte(test.wantInput))
		})
	}
}

func TestRoundTripper(t *testing.T) {
	wantToken := ""

	n := time.Now()
	now := func() time.Time { return n }

	env := []string{""}
	environ := func() []string {
		s := make([]string, len(env))
		copy(s, env)
		return s
	}

	setOutput := func(s string) {
		env[0] = "TEST_OUTPUT=" + s
	}

	handler := func(w http.ResponseWriter, r *http.Request) {
		gotToken := ""
		parts := strings.Split(r.Header.Get("Authorization"), " ")
		if len(parts) > 1 && strings.EqualFold(parts[0], "bearer") {
			gotToken = parts[1]
		}

		if wantToken != gotToken {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}
		fmt.Fprintln(w, "ok")
	}
	server := httptest.NewServer(http.HandlerFunc(handler))

	c := api.ExecConfig{
		Command:    "./testdata/test-plugin.sh",
		APIVersion: "client.authentication.k8s.io/v1alpha1",
	}
	a, err := newAuthenticator(newCache(), &c)
	if err != nil {
		t.Fatal(err)
	}
	a.environ = environ
	a.now = now
	a.stderr = ioutil.Discard

	tc := &transport.Config{}
	if err := a.UpdateTransportConfig(tc); err != nil {
		t.Fatal(err)
	}
	client := http.Client{
		Transport: tc.WrapTransport(http.DefaultTransport),
	}

	get := func(t *testing.T, statusCode int) {
		t.Helper()
		resp, err := client.Get(server.URL)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != statusCode {
			t.Errorf("wanted status %d got %d", statusCode, resp.StatusCode)
		}
	}

	setOutput(`{
		"kind": "ExecCredential",
		"apiVersion": "client.authentication.k8s.io/v1alpha1",
		"status": {
			"token": "token1"
		}
	}`)
	wantToken = "token1"
	get(t, http.StatusOK)

	setOutput(`{
		"kind": "ExecCredential",
		"apiVersion": "client.authentication.k8s.io/v1alpha1",
		"status": {
			"token": "token2"
		}
	}`)
	// Previous token should be cached
	get(t, http.StatusOK)

	wantToken = "token2"
	// Token is still cached, hits unauthorized but causes token to rotate.
	get(t, http.StatusUnauthorized)
	// Follow up request uses the rotated token.
	get(t, http.StatusOK)

	setOutput(`{
		"kind": "ExecCredential",
		"apiVersion": "client.authentication.k8s.io/v1alpha1",
		"status": {
			"token": "token3",
			"expirationTimestamp": "` + now().Add(time.Hour).Format(time.RFC3339Nano) + `"
		}
	}`)
	wantToken = "token3"
	// Token is still cached, hit's unauthorized but causes rotation to token with an expiry.
	get(t, http.StatusUnauthorized)
	get(t, http.StatusOK)

	// Move time forward 2 hours, "token3" is now expired.
	n = n.Add(time.Hour * 2)
	setOutput(`{
		"kind": "ExecCredential",
		"apiVersion": "client.authentication.k8s.io/v1alpha1",
		"status": {
			"token": "token4",
			"expirationTimestamp": "` + now().Add(time.Hour).Format(time.RFC3339Nano) + `"
		}
	}`)
	wantToken = "token4"
	// Old token is expired, should refresh automatically without hitting a 401.
	get(t, http.StatusOK)
}

func TestTLSCredentials(t *testing.T) {
	now := time.Now()

	certPool := x509.NewCertPool()
	cert, key := genClientCert(t)
	if !certPool.AppendCertsFromPEM(cert) {
		t.Fatal("failed to add client cert to CertPool")
	}

	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "ok")
	}))
	server.TLS = &tls.Config{
		ClientAuth: tls.RequireAndVerifyClientCert,
		ClientCAs:  certPool,
	}
	server.StartTLS()
	defer server.Close()

	a, err := newAuthenticator(newCache(), &api.ExecConfig{
		Command:    "./testdata/test-plugin.sh",
		APIVersion: "client.authentication.k8s.io/v1alpha1",
	})
	if err != nil {
		t.Fatal(err)
	}
	var output *clientauthentication.ExecCredential
	a.environ = func() []string {
		data, err := runtime.Encode(codecs.LegacyCodec(a.group), output)
		if err != nil {
			t.Fatal(err)
		}
		return []string{"TEST_OUTPUT=" + string(data)}
	}
	a.now = func() time.Time { return now }
	a.stderr = ioutil.Discard

	// We're not interested in server's cert, this test is about client cert.
	tc := &transport.Config{TLS: transport.TLSConfig{Insecure: true}}
	if err := a.UpdateTransportConfig(tc); err != nil {
		t.Fatal(err)
	}

	get := func(t *testing.T, desc string, wantErr bool) {
		t.Run(desc, func(t *testing.T) {
			tlsCfg, err := transport.TLSConfigFor(tc)
			if err != nil {
				t.Fatal("TLSConfigFor:", err)
			}
			client := http.Client{
				Transport: &http.Transport{TLSClientConfig: tlsCfg},
			}
			resp, err := client.Get(server.URL)
			switch {
			case err != nil && !wantErr:
				t.Errorf("got client.Get error: %q, want nil", err)
			case err == nil && wantErr:
				t.Error("got nil client.Get error, want non-nil")
			}
			if err == nil {
				resp.Body.Close()
			}
		})
	}

	output = &clientauthentication.ExecCredential{
		Status: &clientauthentication.ExecCredentialStatus{
			ClientCertificateData: string(cert),
			ClientKeyData:         string(key),
			ExpirationTimestamp:   &v1.Time{now.Add(time.Hour)},
		},
	}
	get(t, "valid TLS cert", false)

	// Advance time to force re-exec.
	nCert, nKey := genClientCert(t)
	now = now.Add(time.Hour * 2)
	output = &clientauthentication.ExecCredential{
		Status: &clientauthentication.ExecCredentialStatus{
			ClientCertificateData: string(nCert),
			ClientKeyData:         string(nKey),
			ExpirationTimestamp:   &v1.Time{now.Add(time.Hour)},
		},
	}
	get(t, "untrusted TLS cert", true)

	now = now.Add(time.Hour * 2)
	output = &clientauthentication.ExecCredential{
		Status: &clientauthentication.ExecCredentialStatus{
			ClientCertificateData: string(cert),
			ClientKeyData:         string(key),
			ExpirationTimestamp:   &v1.Time{now.Add(time.Hour)},
		},
	}
	get(t, "valid TLS cert again", false)
}

// genClientCert generates an x509 certificate for testing. Certificate and key
// are returned in PEM encoding.
func genClientCert(t *testing.T) ([]byte, []byte) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	keyRaw, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		t.Fatal(err)
	}
	serialNumberLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	serialNumber, err := rand.Int(rand.Reader, serialNumberLimit)
	if err != nil {
		t.Fatal(err)
	}
	cert := &x509.Certificate{
		SerialNumber: serialNumber,
		Subject:      pkix.Name{Organization: []string{"Acme Co"}},
		NotBefore:    time.Now(),
		NotAfter:     time.Now().Add(24 * time.Hour),

		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
		BasicConstraintsValid: true,
	}
	certRaw, err := x509.CreateCertificate(rand.Reader, cert, cert, key.Public(), key)
	if err != nil {
		t.Fatal(err)
	}
	return pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certRaw}),
		pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyRaw})
}
