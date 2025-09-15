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
	"io"
	"math/big"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/pkg/apis/clientauthentication"
	"k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/client-go/transport"
	testingclock "k8s.io/utils/clock/testing"
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
	cert.Leaf, err = x509.ParseCertificate(cert.Certificate[0])
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
		APIVersion:         "client.authentication.k8s.io/v1beta1",
		ProvideClusterInfo: true,
	}
	c1c := &clientauthentication.Cluster{
		Server:                   "foo",
		TLSServerName:            "bar",
		CertificateAuthorityData: []byte("baz"),
		Config: &runtime.Unknown{
			TypeMeta: runtime.TypeMeta{
				APIVersion: "",
				Kind:       "",
			},
			Raw:             []byte(`{"apiVersion":"group/v1","kind":"PluginConfig","spec":{"audience":"snorlax"}}`),
			ContentEncoding: "",
			ContentType:     "application/json",
		},
	}

	c2 := &api.ExecConfig{
		Command: "foo-bar",
		Args:    []string{"1", "2"},
		Env: []api.ExecEnvVar{
			{Name: "3", Value: "4"},
			{Name: "5", Value: "6"},
			{Name: "7", Value: "8"},
		},
		APIVersion:         "client.authentication.k8s.io/v1beta1",
		ProvideClusterInfo: true,
	}
	c2c := &clientauthentication.Cluster{
		Server:                   "foo",
		TLSServerName:            "bar",
		CertificateAuthorityData: []byte("baz"),
		Config: &runtime.Unknown{
			TypeMeta: runtime.TypeMeta{
				APIVersion: "",
				Kind:       "",
			},
			Raw:             []byte(`{"apiVersion":"group/v1","kind":"PluginConfig","spec":{"audience":"snorlax"}}`),
			ContentEncoding: "",
			ContentType:     "application/json",
		},
	}

	c3 := &api.ExecConfig{
		Command: "foo-bar",
		Args:    []string{"1", "2"},
		Env: []api.ExecEnvVar{
			{Name: "3", Value: "4"},
			{Name: "5", Value: "6"},
		},
		APIVersion: "client.authentication.k8s.io/v1beta1",
	}
	c3c := &clientauthentication.Cluster{
		Server:                   "foo",
		TLSServerName:            "bar",
		CertificateAuthorityData: []byte("baz"),
		Config: &runtime.Unknown{
			TypeMeta: runtime.TypeMeta{
				APIVersion: "",
				Kind:       "",
			},
			Raw:             []byte(`{"apiVersion":"group/v1","kind":"PluginConfig","spec":{"audience":"snorlax"}}`),
			ContentEncoding: "",
			ContentType:     "application/json",
		},
	}

	c4 := &api.ExecConfig{
		Command: "foo-bar",
		Args:    []string{"1", "2"},
		Env: []api.ExecEnvVar{
			{Name: "3", Value: "4"},
			{Name: "5", Value: "6"},
		},
		APIVersion: "client.authentication.k8s.io/v1beta1",
	}
	c4c := &clientauthentication.Cluster{
		Server:                   "foo",
		TLSServerName:            "bar",
		CertificateAuthorityData: []byte("baz"),
		Config: &runtime.Unknown{
			TypeMeta: runtime.TypeMeta{
				APIVersion: "",
				Kind:       "",
			},
			Raw:             []byte(`{"apiVersion":"group/v1","kind":"PluginConfig","spec":{"audience":"panda"}}`),
			ContentEncoding: "",
			ContentType:     "application/json",
		},
	}

	// c5/c5c should be the same as c4/c4c, except c5 has ProvideClusterInfo set to true.
	c5 := &api.ExecConfig{
		Command: "foo-bar",
		Args:    []string{"1", "2"},
		Env: []api.ExecEnvVar{
			{Name: "3", Value: "4"},
			{Name: "5", Value: "6"},
		},
		APIVersion:         "client.authentication.k8s.io/v1beta1",
		ProvideClusterInfo: true,
	}
	c5c := &clientauthentication.Cluster{
		Server:                   "foo",
		TLSServerName:            "bar",
		CertificateAuthorityData: []byte("baz"),
		Config: &runtime.Unknown{
			TypeMeta: runtime.TypeMeta{
				APIVersion: "",
				Kind:       "",
			},
			Raw:             []byte(`{"apiVersion":"group/v1","kind":"PluginConfig","spec":{"audience":"panda"}}`),
			ContentEncoding: "",
			ContentType:     "application/json",
		},
	}

	// c6 should be the same as c4, except c6 is passed with a nil cluster
	c6 := &api.ExecConfig{
		Command: "foo-bar",
		Args:    []string{"1", "2"},
		Env: []api.ExecEnvVar{
			{Name: "3", Value: "4"},
			{Name: "5", Value: "6"},
		},
		APIVersion: "client.authentication.k8s.io/v1betaa1",
	}

	// c7 should be the same as c6, except c7 has stdin marked as unavailable
	c7 := &api.ExecConfig{
		Command: "foo-bar",
		Args:    []string{"1", "2"},
		Env: []api.ExecEnvVar{
			{Name: "3", Value: "4"},
			{Name: "5", Value: "6"},
		},
		APIVersion:       "client.authentication.k8s.io/v1beta1",
		StdinUnavailable: true,
	}

	key1 := cacheKey(c1, c1c)
	key2 := cacheKey(c2, c2c)
	key3 := cacheKey(c3, c3c)
	key4 := cacheKey(c4, c4c)
	key5 := cacheKey(c5, c5c)
	key6 := cacheKey(c6, nil)
	key7 := cacheKey(c7, nil)
	if key1 != key2 {
		t.Error("key1 and key2 didn't match")
	}
	if key1 == key3 {
		t.Error("key1 and key3 matched")
	}
	if key2 == key3 {
		t.Error("key2 and key3 matched")
	}
	if key3 == key4 {
		t.Error("key3 and key4 matched")
	}
	if key4 == key5 {
		t.Error("key3 and key4 matched")
	}
	if key6 == key4 {
		t.Error("key6 and key4 matched")
	}
	if key6 == key7 {
		t.Error("key6 and key7 matched")
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
		name             string
		config           api.ExecConfig
		stdinUnavailable bool
		exitCode         int
		cluster          *clientauthentication.Cluster
		output           string
		isTerminal       bool
		wantInput        string
		wantCreds        credentials
		wantExpiry       time.Time
		wantErr          bool
		wantErrSubstr    string
	}{
		{
			name: "beta-with-TLS-credentials",
			config: api.ExecConfig{
				APIVersion:      "client.authentication.k8s.io/v1beta1",
				InteractiveMode: api.IfAvailableExecInteractiveMode,
			},
			wantInput: `{
				"kind":"ExecCredential",
				"apiVersion":"client.authentication.k8s.io/v1beta1",
				"spec": {
					"interactive": false
				}
			}`,
			output: fmt.Sprintf(`{
				"kind": "ExecCredential",
				"apiVersion": "client.authentication.k8s.io/v1beta1",
				"status": {
					"clientKeyData": %q,
					"clientCertificateData": %q
				}
			}`, keyData, certData),
			wantCreds: credentials{cert: validCert},
		},
		{
			name: "beta-with-bad-TLS-credentials",
			config: api.ExecConfig{
				APIVersion:      "client.authentication.k8s.io/v1beta1",
				InteractiveMode: api.IfAvailableExecInteractiveMode,
			},
			output: `{
				"kind": "ExecCredential",
				"apiVersion": "client.authentication.k8s.io/v1beta1",
				"status": {
					"clientKeyData": "foo",
					"clientCertificateData": "bar"
				}
			}`,
			wantErr: true,
		},
		{
			name: "beta-cert-but-no-key",
			config: api.ExecConfig{
				APIVersion:      "client.authentication.k8s.io/v1beta1",
				InteractiveMode: api.IfAvailableExecInteractiveMode,
			},
			output: fmt.Sprintf(`{
				"kind": "ExecCredential",
				"apiVersion": "client.authentication.k8s.io/v1beta1",
				"status": {
					"clientCertificateData": %q
				}
			}`, certData),
			wantErr: true,
		},
		{
			name: "beta-basic-request",
			config: api.ExecConfig{
				APIVersion:      "client.authentication.k8s.io/v1beta1",
				InteractiveMode: api.IfAvailableExecInteractiveMode,
			},
			wantInput: `{
				"kind": "ExecCredential",
				"apiVersion": "client.authentication.k8s.io/v1beta1",
				"spec": {
					"interactive": false
				}
			}`,
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
			name: "beta-basic-request-with-never-interactive-mode",
			config: api.ExecConfig{
				APIVersion:      "client.authentication.k8s.io/v1beta1",
				InteractiveMode: api.NeverExecInteractiveMode,
			},
			wantInput: `{
				"kind": "ExecCredential",
				"apiVersion": "client.authentication.k8s.io/v1beta1",
				"spec": {
					"interactive": false
				}
			}`,
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
			name: "beta-basic-request-with-never-interactive-mode-and-stdin-unavailable",
			config: api.ExecConfig{
				APIVersion:       "client.authentication.k8s.io/v1beta1",
				InteractiveMode:  api.NeverExecInteractiveMode,
				StdinUnavailable: true,
			},
			wantInput: `{
				"kind": "ExecCredential",
				"apiVersion": "client.authentication.k8s.io/v1beta1",
				"spec": {
					"interactive": false
				}
			}`,
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
			name: "beta-basic-request-with-if-available-interactive-mode",
			config: api.ExecConfig{
				APIVersion:      "client.authentication.k8s.io/v1beta1",
				InteractiveMode: api.IfAvailableExecInteractiveMode,
			},
			wantInput: `{
				"kind": "ExecCredential",
				"apiVersion": "client.authentication.k8s.io/v1beta1",
				"spec": {
					"interactive": false
				}
			}`,
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
			name: "beta-basic-request-with-if-available-interactive-mode-and-stdin-unavailable",
			config: api.ExecConfig{
				APIVersion:       "client.authentication.k8s.io/v1beta1",
				InteractiveMode:  api.IfAvailableExecInteractiveMode,
				StdinUnavailable: true,
			},
			wantInput: `{
				"kind": "ExecCredential",
				"apiVersion": "client.authentication.k8s.io/v1beta1",
				"spec": {
					"interactive": false
				}
			}`,
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
			name: "beta-basic-request-with-if-available-interactive-mode-and-terminal",
			config: api.ExecConfig{
				APIVersion:      "client.authentication.k8s.io/v1beta1",
				InteractiveMode: api.IfAvailableExecInteractiveMode,
			},
			isTerminal: true,
			wantInput: `{
				"kind": "ExecCredential",
				"apiVersion": "client.authentication.k8s.io/v1beta1",
				"spec": {
					"interactive": true
				}
			}`,
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
			name: "beta-basic-request-with-if-available-interactive-mode-and-terminal-and-stdin-unavailable",
			config: api.ExecConfig{
				APIVersion:       "client.authentication.k8s.io/v1beta1",
				InteractiveMode:  api.IfAvailableExecInteractiveMode,
				StdinUnavailable: true,
			},
			isTerminal: true,
			wantInput: `{
				"kind": "ExecCredential",
				"apiVersion": "client.authentication.k8s.io/v1beta1",
				"spec": {
					"interactive": false
				}
			}`,
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
			name: "beta-basic-request-with-always-interactive-mode",
			config: api.ExecConfig{
				APIVersion:      "client.authentication.k8s.io/v1beta1",
				InteractiveMode: api.AlwaysExecInteractiveMode,
			},
			wantErr:       true,
			wantErrSubstr: "exec plugin cannot support interactive mode: standard input is not a terminal",
		},
		{
			name: "beta-basic-request-with-always-interactive-mode-and-terminal-and-stdin-unavailable",
			config: api.ExecConfig{
				APIVersion:       "client.authentication.k8s.io/v1beta1",
				InteractiveMode:  api.AlwaysExecInteractiveMode,
				StdinUnavailable: true,
			},
			isTerminal:    true,
			wantErr:       true,
			wantErrSubstr: "exec plugin cannot support interactive mode: standard input is unavailable",
		},
		{
			name: "beta-basic-request-with-always-interactive-mode-and-terminal-and-stdin-unavailable-with-message",
			config: api.ExecConfig{
				APIVersion:              "client.authentication.k8s.io/v1beta1",
				InteractiveMode:         api.AlwaysExecInteractiveMode,
				StdinUnavailable:        true,
				StdinUnavailableMessage: "some message",
			},
			isTerminal:    true,
			wantErr:       true,
			wantErrSubstr: "exec plugin cannot support interactive mode: standard input is unavailable: some message",
		},
		{
			name: "beta-basic-request-with-always-interactive-mode-and-terminal",
			config: api.ExecConfig{
				APIVersion:      "client.authentication.k8s.io/v1beta1",
				InteractiveMode: api.AlwaysExecInteractiveMode,
			},
			isTerminal: true,
			wantInput: `{
				"kind": "ExecCredential",
				"apiVersion": "client.authentication.k8s.io/v1beta1",
				"spec": {
					"interactive": true
				}
			}`,
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
				APIVersion:      "client.authentication.k8s.io/v1beta1",
				InteractiveMode: api.IfAvailableExecInteractiveMode,
			},
			wantInput: `{
				"kind": "ExecCredential",
				"apiVersion": "client.authentication.k8s.io/v1beta1",
				"spec": {
					"interactive": false
				}
			}`,
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
				APIVersion:      "client.authentication.k8s.io/v1beta1",
				InteractiveMode: api.IfAvailableExecInteractiveMode,
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
				APIVersion:      "client.authentication.k8s.io/v1beta1",
				InteractiveMode: api.IfAvailableExecInteractiveMode,
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
				APIVersion:      "client.authentication.k8s.io/v1beta1",
				InteractiveMode: api.IfAvailableExecInteractiveMode,
			},
			output: `{
				"kind": "ExecCredential",
				"apiVersion":"client.authentication.k8s.io/v1beta1",
				"status": {}
			}`,
			wantErr: true,
		},
		{
			name: "unknown-binary",
			config: api.ExecConfig{
				APIVersion:      "client.authentication.k8s.io/v1beta1",
				Command:         "does not exist",
				InstallHint:     "some install hint",
				InteractiveMode: api.IfAvailableExecInteractiveMode,
			},
			wantErr:       true,
			wantErrSubstr: "some install hint",
		},
		{
			name: "binary-fails",
			config: api.ExecConfig{
				APIVersion:      "client.authentication.k8s.io/v1beta1",
				InteractiveMode: api.IfAvailableExecInteractiveMode,
			},
			exitCode:      73,
			wantErr:       true,
			wantErrSubstr: "73",
		},
		{
			name: "beta-with-cluster-and-provide-cluster-info-is-serialized",
			config: api.ExecConfig{
				APIVersion:         "client.authentication.k8s.io/v1beta1",
				ProvideClusterInfo: true,
				InteractiveMode:    api.IfAvailableExecInteractiveMode,
			},
			cluster: &clientauthentication.Cluster{
				Server:                   "foo",
				TLSServerName:            "bar",
				CertificateAuthorityData: []byte("baz"),
				Config: &runtime.Unknown{
					TypeMeta: runtime.TypeMeta{
						APIVersion: "",
						Kind:       "",
					},
					Raw:             []byte(`{"apiVersion":"group/v1","kind":"PluginConfig","spec":{"audience":"snorlax"}}`),
					ContentEncoding: "",
					ContentType:     "application/json",
				},
			},
			wantInput: `{
				"kind":"ExecCredential",
				"apiVersion":"client.authentication.k8s.io/v1beta1",
				"spec": {
					"cluster": {
						"server": "foo",
						"tls-server-name": "bar",
						"certificate-authority-data": "YmF6",
						"config": {
							"apiVersion": "group/v1",
							"kind": "PluginConfig",
							"spec": {
								"audience": "snorlax"
							}
						}
					},
					"interactive": false
				}
			}`,
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
			name: "beta-with-cluster-and-without-provide-cluster-info-is-not-serialized",
			config: api.ExecConfig{
				APIVersion:      "client.authentication.k8s.io/v1beta1",
				InteractiveMode: api.IfAvailableExecInteractiveMode,
			},
			cluster: &clientauthentication.Cluster{
				Server:                   "foo",
				TLSServerName:            "bar",
				CertificateAuthorityData: []byte("baz"),
				Config: &runtime.Unknown{
					TypeMeta: runtime.TypeMeta{
						APIVersion: "",
						Kind:       "",
					},
					Raw:             []byte(`{"apiVersion":"group/v1","kind":"PluginConfig","spec":{"audience":"snorlax"}}`),
					ContentEncoding: "",
					ContentType:     "application/json",
				},
			},
			wantInput: `{
				"kind":"ExecCredential",
				"apiVersion":"client.authentication.k8s.io/v1beta1",
				"spec": {
					"interactive": false
				}
			}`,
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
			name: "v1-basic-request",
			config: api.ExecConfig{
				APIVersion:      "client.authentication.k8s.io/v1",
				InteractiveMode: api.IfAvailableExecInteractiveMode,
			},
			wantInput: `{
				"kind": "ExecCredential",
				"apiVersion": "client.authentication.k8s.io/v1",
				"spec": {
					"interactive": false
				}
			}`,
			output: `{
				"kind": "ExecCredential",
				"apiVersion": "client.authentication.k8s.io/v1",
				"status": {
					"token": "foo-bar"
				}
			}`,
			wantCreds: credentials{token: "foo-bar"},
		},
		{
			name: "v1-with-missing-interactive-mode",
			config: api.ExecConfig{
				APIVersion: "client.authentication.k8s.io/v1",
			},
			wantErr:       true,
			wantErrSubstr: `exec plugin cannot support interactive mode: unknown interactiveMode: ""`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			c := test.config

			if c.Command == "" {
				c.Command = "./testdata/test-plugin.sh"
				c.Env = append(c.Env, api.ExecEnvVar{
					Name:  "TEST_OUTPUT",
					Value: test.output,
				})
				c.Env = append(c.Env, api.ExecEnvVar{
					Name:  "TEST_EXIT_CODE",
					Value: strconv.Itoa(test.exitCode),
				})
			}

			a, err := newAuthenticator(newCache(), func(_ int) bool { return test.isTerminal }, &c, test.cluster)
			if err != nil {
				t.Fatal(err)
			}

			stderr := &bytes.Buffer{}
			a.stderr = stderr
			a.environ = func() []string { return nil }

			if err := a.refreshCredsLocked(); err != nil {
				if !test.wantErr {
					t.Errorf("get token %v", err)
				} else if !strings.Contains(err.Error(), test.wantErrSubstr) {
					t.Errorf("expected error with substring '%v' got '%v'", test.wantErrSubstr, err.Error())
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
		Command:         "./testdata/test-plugin.sh",
		APIVersion:      "client.authentication.k8s.io/v1beta1",
		InteractiveMode: api.IfAvailableExecInteractiveMode,
	}
	a, err := newAuthenticator(newCache(), func(_ int) bool { return false }, &c, nil)
	if err != nil {
		t.Fatal(err)
	}
	a.environ = environ
	a.now = now
	a.stderr = io.Discard

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
		"apiVersion": "client.authentication.k8s.io/v1beta1",
		"status": {
			"token": "token1"
		}
	}`)
	wantToken = "token1"
	get(t, http.StatusOK)

	setOutput(`{
		"kind": "ExecCredential",
		"apiVersion": "client.authentication.k8s.io/v1beta1",
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
		"apiVersion": "client.authentication.k8s.io/v1beta1",
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
		"apiVersion": "client.authentication.k8s.io/v1beta1",
		"status": {
			"token": "token4",
			"expirationTimestamp": "` + now().Add(time.Hour).Format(time.RFC3339Nano) + `"
		}
	}`)
	wantToken = "token4"
	// Old token is expired, should refresh automatically without hitting a 401.
	get(t, http.StatusOK)
}

func TestAuthorizationHeaderPresentCancelsExecAction(t *testing.T) {
	tests := []struct {
		name               string
		setTransportConfig func(*transport.Config)
	}{
		{
			name: "bearer token",
			setTransportConfig: func(config *transport.Config) {
				config.BearerToken = "token1f"
			},
		},
		{
			name: "basic auth",
			setTransportConfig: func(config *transport.Config) {
				config.Username = "marshmallow"
				config.Password = "zelda"
			},
		},
		{
			name: "cert auth",
			setTransportConfig: func(config *transport.Config) {
				config.TLS.CertData = []byte("some-cert-data")
				config.TLS.KeyData = []byte("some-key-data")
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a, err := newAuthenticator(newCache(), func(_ int) bool { return false }, &api.ExecConfig{
				Command:    "./testdata/test-plugin.sh",
				APIVersion: "client.authentication.k8s.io/v1beta1",
			}, nil)
			if err != nil {
				t.Fatal(err)
			}

			// UpdateTransportConfig returns error on existing TLS certificate callback, unless a bearer token is present in the
			// transport config, in which case it takes precedence
			cert := func() (*tls.Certificate, error) {
				return nil, nil
			}
			tc := &transport.Config{TLS: transport.TLSConfig{Insecure: true, GetCertHolder: &transport.GetCertHolder{GetCert: cert}}}
			test.setTransportConfig(tc)

			if err := a.UpdateTransportConfig(tc); err != nil {
				t.Error("Expected presence of bearer token in config to cancel exec action")
			}
		})
	}
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

	a, err := newAuthenticator(newCache(), func(_ int) bool { return false }, &api.ExecConfig{
		Command:         "./testdata/test-plugin.sh",
		APIVersion:      "client.authentication.k8s.io/v1beta1",
		InteractiveMode: api.IfAvailableExecInteractiveMode,
	}, nil)
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
	a.stderr = io.Discard

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
			ExpirationTimestamp:   &v1.Time{Time: now.Add(time.Hour)},
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
			ExpirationTimestamp:   &v1.Time{Time: now.Add(time.Hour)},
		},
	}
	get(t, "untrusted TLS cert", true)

	now = now.Add(time.Hour * 2)
	output = &clientauthentication.ExecCredential{
		Status: &clientauthentication.ExecCredentialStatus{
			ClientCertificateData: string(cert),
			ClientKeyData:         string(key),
			ExpirationTimestamp:   &v1.Time{Time: now.Add(time.Hour)},
		},
	}
	get(t, "valid TLS cert again", false)
}

func TestConcurrentUpdateTransportConfig(t *testing.T) {
	n := time.Now()
	now := func() time.Time { return n }

	env := []string{""}
	environ := func() []string {
		s := make([]string, len(env))
		copy(s, env)
		return s
	}

	c := api.ExecConfig{
		Command:    "./testdata/test-plugin.sh",
		APIVersion: "client.authentication.k8s.io/v1beta1",
	}
	a, err := newAuthenticator(newCache(), func(_ int) bool { return false }, &c, nil)
	if err != nil {
		t.Fatal(err)
	}
	a.environ = environ
	a.now = now
	a.stderr = io.Discard

	stopCh := make(chan struct{})
	defer close(stopCh)

	numConcurrent := 2

	for i := 0; i < numConcurrent; i++ {
		go func() {
			for {
				tc := &transport.Config{}
				a.UpdateTransportConfig(tc)

				select {
				case <-stopCh:
					return
				default:
					continue
				}
			}
		}()
	}
	time.Sleep(2 * time.Second)
}

func TestInstallHintRateLimit(t *testing.T) {
	tests := []struct {
		name string

		threshold int
		interval  time.Duration

		calls          int
		perCallAdvance time.Duration

		wantInstallHint int
	}{
		{
			name:            "print-up-to-threshold",
			threshold:       2,
			interval:        time.Second,
			calls:           10,
			wantInstallHint: 2,
		},
		{
			name:            "after-interval-threshold-resets",
			threshold:       2,
			interval:        time.Second * 5,
			calls:           10,
			perCallAdvance:  time.Second,
			wantInstallHint: 4,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			c := api.ExecConfig{
				Command:         "does not exist",
				APIVersion:      "client.authentication.k8s.io/v1beta1",
				InstallHint:     "some install hint",
				InteractiveMode: api.IfAvailableExecInteractiveMode,
			}
			a, err := newAuthenticator(newCache(), func(_ int) bool { return false }, &c, nil)
			if err != nil {
				t.Fatal(err)
			}

			a.sometimes.threshold = test.threshold
			a.sometimes.interval = test.interval

			clock := testingclock.NewFakeClock(time.Now())
			a.sometimes.clock = clock

			count := 0
			for i := 0; i < test.calls; i++ {
				err := a.refreshCredsLocked()
				if strings.Contains(err.Error(), c.InstallHint) {
					count++
				}

				clock.SetTime(clock.Now().Add(test.perCallAdvance))
			}

			if test.wantInstallHint != count {
				t.Errorf(
					"%s: expected install hint %d times got %d",
					test.name,
					test.wantInstallHint,
					count,
				)
			}
		})
	}
}

// genClientCert generates an x509 certificate for testing. Certificate and key
// are returned in PEM encoding. The generated cert expires in 24 hours.
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
