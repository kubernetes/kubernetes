/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package client

import (
	"bytes"
	"encoding/json"
	"io"
	"io/ioutil"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
)

const (
	rootCACert = `-----BEGIN CERTIFICATE-----
MIIC4DCCAcqgAwIBAgIBATALBgkqhkiG9w0BAQswIzEhMB8GA1UEAwwYMTAuMTMu
MTI5LjEwNkAxNDIxMzU5MDU4MB4XDTE1MDExNTIxNTczN1oXDTE2MDExNTIxNTcz
OFowIzEhMB8GA1UEAwwYMTAuMTMuMTI5LjEwNkAxNDIxMzU5MDU4MIIBIjANBgkq
hkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAunDRXGwsiYWGFDlWH6kjGun+PshDGeZX
xtx9lUnL8pIRWH3wX6f13PO9sktaOWW0T0mlo6k2bMlSLlSZgG9H6og0W6gLS3vq
s4VavZ6DbXIwemZG2vbRwsvR+t4G6Nbwelm6F8RFnA1Fwt428pavmNQ/wgYzo+T1
1eS+HiN4ACnSoDSx3QRWcgBkB1g6VReofVjx63i0J+w8Q/41L9GUuLqquFxu6ZnH
60vTB55lHgFiDLjA1FkEz2dGvGh/wtnFlRvjaPC54JH2K1mPYAUXTreoeJtLJKX0
ycoiyB24+zGCniUmgIsmQWRPaOPircexCp1BOeze82BT1LCZNTVaxQIDAQABoyMw
ITAOBgNVHQ8BAf8EBAMCAKQwDwYDVR0TAQH/BAUwAwEB/zALBgkqhkiG9w0BAQsD
ggEBADMxsUuAFlsYDpF4fRCzXXwrhbtj4oQwcHpbu+rnOPHCZupiafzZpDu+rw4x
YGPnCb594bRTQn4pAu3Ac18NbLD5pV3uioAkv8oPkgr8aUhXqiv7KdDiaWm6sbAL
EHiXVBBAFvQws10HMqMoKtO8f1XDNAUkWduakR/U6yMgvOPwS7xl0eUTqyRB6zGb
K55q2dejiFWaFqB/y78txzvz6UlOZKE44g2JAVoJVM6kGaxh33q8/FmrL4kuN3ut
W+MmJCVDvd4eEqPwbp7146ZWTqpIJ8lvA6wuChtqV8lhAPka2hD/LMqY8iXNmfXD
uml0obOEy+ON91k+SWTJ3ggmF/U=
-----END CERTIFICATE-----`

	certData = `-----BEGIN CERTIFICATE-----
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
-----END CERTIFICATE-----`

	keyData = `-----BEGIN RSA PRIVATE KEY-----
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
-----END RSA PRIVATE KEY-----`
)

func TestTransportFor(t *testing.T) {
	testCases := map[string]struct {
		Config  *Config
		Err     bool
		TLS     bool
		Default bool
	}{
		"default transport": {
			Default: true,
			Config:  &Config{},
		},

		"ca transport": {
			TLS: true,
			Config: &Config{
				TLSClientConfig: TLSClientConfig{
					CAData: []byte(rootCACert),
				},
			},
		},
		"bad ca file transport": {
			Err: true,
			Config: &Config{
				TLSClientConfig: TLSClientConfig{
					CAFile: "invalid file",
				},
			},
		},
		"ca data overriding bad ca file transport": {
			TLS: true,
			Config: &Config{
				TLSClientConfig: TLSClientConfig{
					CAData: []byte(rootCACert),
					CAFile: "invalid file",
				},
			},
		},

		"cert transport": {
			TLS: true,
			Config: &Config{
				TLSClientConfig: TLSClientConfig{
					CertData: []byte(certData),
					KeyData:  []byte(keyData),
					CAData:   []byte(rootCACert),
				},
			},
		},
		"bad cert data transport": {
			Err: true,
			Config: &Config{
				TLSClientConfig: TLSClientConfig{
					CertData: []byte(certData),
					KeyData:  []byte("bad key data"),
					CAData:   []byte(rootCACert),
				},
			},
		},
		"bad file cert transport": {
			Err: true,
			Config: &Config{
				TLSClientConfig: TLSClientConfig{
					CertData: []byte(certData),
					KeyFile:  "invalid file",
					CAData:   []byte(rootCACert),
				},
			},
		},
		"key data overriding bad file cert transport": {
			TLS: true,
			Config: &Config{
				TLSClientConfig: TLSClientConfig{
					CertData: []byte(certData),
					KeyData:  []byte(keyData),
					KeyFile:  "invalid file",
					CAData:   []byte(rootCACert),
				},
			},
		},
	}
	for k, testCase := range testCases {
		transport, err := TransportFor(testCase.Config)
		switch {
		case testCase.Err && err == nil:
			t.Errorf("%s: unexpected non-error", k)
			continue
		case !testCase.Err && err != nil:
			t.Errorf("%s: unexpected error: %v", k, err)
			continue
		}

		switch {
		case testCase.Default && transport != http.DefaultTransport:
			t.Errorf("%s: expected the default transport, got %#v", k, transport)
			continue
		case !testCase.Default && transport == http.DefaultTransport:
			t.Errorf("%s: expected non-default transport, got %#v", k, transport)
			continue
		}

		// We only know how to check TLSConfig on http.Transports
		if transport, ok := transport.(*http.Transport); ok {
			switch {
			case testCase.TLS && transport.TLSClientConfig == nil:
				t.Errorf("%s: expected TLSClientConfig, got %#v", k, transport)
				continue
			case !testCase.TLS && transport.TLSClientConfig != nil:
				t.Errorf("%s: expected no TLSClientConfig, got %#v", k, transport)
				continue
			}
		}
	}
}

func TestIsConfigTransportTLS(t *testing.T) {
	testCases := []struct {
		Config       *Config
		TransportTLS bool
	}{
		{
			Config:       &Config{},
			TransportTLS: false,
		},
		{
			Config: &Config{
				Host: "https://localhost",
			},
			TransportTLS: true,
		},
		{
			Config: &Config{
				Host: "localhost",
				TLSClientConfig: TLSClientConfig{
					CertFile: "foo",
				},
			},
			TransportTLS: true,
		},
		{
			Config: &Config{
				Host: "///:://localhost",
				TLSClientConfig: TLSClientConfig{
					CertFile: "foo",
				},
			},
			TransportTLS: false,
		},
		{
			Config: &Config{
				Host:     "1.2.3.4:567",
				Insecure: true,
			},
			TransportTLS: true,
		},
	}
	for _, testCase := range testCases {
		if err := SetKubernetesDefaults(testCase.Config); err != nil {
			t.Errorf("setting defaults failed for %#v: %v", testCase.Config, err)
			continue
		}
		useTLS := IsConfigTransportTLS(*testCase.Config)
		if testCase.TransportTLS != useTLS {
			t.Errorf("expected %v for %#v", testCase.TransportTLS, testCase.Config)
		}
	}
}

func TestSetKubernetesDefaults(t *testing.T) {
	testCases := []struct {
		Config Config
		After  Config
		Err    bool
	}{
		{
			Config{},
			Config{
				Prefix:  "/api",
				Version: latest.Version,
				Codec:   latest.Codec,
				QPS:     5,
				Burst:   10,
			},
			false,
		},
		{
			Config{
				Version: "not_an_api",
			},
			Config{},
			true,
		},
	}
	for _, testCase := range testCases {
		val := &testCase.Config
		err := SetKubernetesDefaults(val)
		val.UserAgent = ""
		switch {
		case err == nil && testCase.Err:
			t.Errorf("expected error but was nil")
			continue
		case err != nil && !testCase.Err:
			t.Errorf("unexpected error %v", err)
			continue
		case err != nil:
			continue
		}
		if !reflect.DeepEqual(*val, testCase.After) {
			t.Errorf("unexpected result object: %#v", val)
		}
	}
}

func TestSetKubernetesDefaultsUserAgent(t *testing.T) {
	config := &Config{}
	if err := SetKubernetesDefaults(config); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(config.UserAgent, "kubernetes/") {
		t.Errorf("no user agent set: %#v", config)
	}
}

func objBody(object interface{}) io.ReadCloser {
	output, err := json.MarshalIndent(object, "", "")
	if err != nil {
		panic(err)
	}
	return ioutil.NopCloser(bytes.NewReader([]byte(output)))
}

func TestNegotiateVersion(t *testing.T) {
	tests := []struct {
		name, version, expectedVersion string
		serverVersions                 []string
		clientVersions                 []string
		config                         *Config
		expectErr                      bool
	}{
		{
			name:            "server supports client default",
			version:         "version1",
			config:          &Config{},
			serverVersions:  []string{"version1", testapi.Version()},
			clientVersions:  []string{"version1", testapi.Version()},
			expectedVersion: "version1",
			expectErr:       false,
		},
		{
			name:            "server falls back to client supported",
			version:         testapi.Version(),
			config:          &Config{},
			serverVersions:  []string{"version1"},
			clientVersions:  []string{"version1", testapi.Version()},
			expectedVersion: "version1",
			expectErr:       false,
		},
		{
			name:            "explicit version supported",
			version:         "",
			config:          &Config{Version: testapi.Version()},
			serverVersions:  []string{"version1", testapi.Version()},
			clientVersions:  []string{"version1", testapi.Version()},
			expectedVersion: testapi.Version(),
			expectErr:       false,
		},
		{
			name:            "explicit version not supported",
			version:         "",
			config:          &Config{Version: testapi.Version()},
			serverVersions:  []string{"version1"},
			clientVersions:  []string{"version1", testapi.Version()},
			expectedVersion: "",
			expectErr:       true,
		},
	}
	codec := testapi.Codec()

	for _, test := range tests {
		fakeClient := &FakeRESTClient{
			Codec: codec,
			Resp: &http.Response{
				StatusCode: 200,
				Body:       objBody(&api.APIVersions{Versions: test.serverVersions}),
			},
			Client: HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: 200, Body: objBody(&api.APIVersions{Versions: test.serverVersions})}, nil
			}),
		}
		c := NewOrDie(test.config)
		c.Client = fakeClient.Client
		response, err := NegotiateVersion(c, test.config, test.version, test.clientVersions)
		if err == nil && test.expectErr {
			t.Errorf("expected error, got nil for [%s].", test.name)
		}
		if err != nil && !test.expectErr {
			t.Errorf("unexpected error for [%s]: %v.", test.name, err)
		}
		if response != test.expectedVersion {
			t.Errorf("expected version %s, got %s.", test.expectedVersion, response)
		}
	}
}
