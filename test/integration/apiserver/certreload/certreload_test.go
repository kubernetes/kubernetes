/*
Copyright 2019 The Kubernetes Authors.

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

package podlogs

import (
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"io/ioutil"
	"net/url"
	"path"
	"strings"
	"testing"
	"time"

	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestClientCA(t *testing.T) {
	stopCh := make(chan struct{})
	defer close(stopCh)

	// I have no idea what this cert is, but it doesn't matter, we just want something that always fails validation
	differentClientCA := []byte(`-----BEGIN CERTIFICATE-----
MIIDQDCCAiigAwIBAgIJANWw74P5KJk2MA0GCSqGSIb3DQEBCwUAMDQxMjAwBgNV
BAMMKWdlbmVyaWNfd2ViaG9va19hZG1pc3Npb25fcGx1Z2luX3Rlc3RzX2NhMCAX
DTE3MTExNjAwMDUzOVoYDzIyOTEwOTAxMDAwNTM5WjAjMSEwHwYDVQQDExh3ZWJo
b29rLXRlc3QuZGVmYXVsdC5zdmMwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEK
AoIBAQDXd/nQ89a5H8ifEsigmMd01Ib6NVR3bkJjtkvYnTbdfYEBj7UzqOQtHoLa
dIVmefny5uIHvj93WD8WDVPB3jX2JHrXkDTXd/6o6jIXHcsUfFTVLp6/bZ+Anqe0
r/7hAPkzA2A7APyTWM3ZbEeo1afXogXhOJ1u/wz0DflgcB21gNho4kKTONXO3NHD
XLpspFqSkxfEfKVDJaYAoMnYZJtFNsa2OvsmLnhYF8bjeT3i07lfwrhUZvP+7Gsp
7UgUwc06WuNHjfx1s5e6ySzH0QioMD1rjYneqOvk0pKrMIhuAEWXqq7jlXcDtx1E
j+wnYbVqqVYheHZ8BCJoVAAQGs9/AgMBAAGjZDBiMAkGA1UdEwQCMAAwCwYDVR0P
BAQDAgXgMB0GA1UdJQQWMBQGCCsGAQUFBwMCBggrBgEFBQcDATApBgNVHREEIjAg
hwR/AAABghh3ZWJob29rLXRlc3QuZGVmYXVsdC5zdmMwDQYJKoZIhvcNAQELBQAD
ggEBAD/GKSPNyQuAOw/jsYZesb+RMedbkzs18sSwlxAJQMUrrXwlVdHrA8q5WhE6
ABLqU1b8lQ8AWun07R8k5tqTmNvCARrAPRUqls/ryER+3Y9YEcxEaTc3jKNZFLbc
T6YtcnkdhxsiO136wtiuatpYL91RgCmuSpR8+7jEHhuFU01iaASu7ypFrUzrKHTF
bKwiLRQi1cMzVcLErq5CDEKiKhUkoDucyARFszrGt9vNIl/YCcBOkcNvM3c05Hn3
M++C29JwS3Hwbubg6WO3wjFjoEhpCwU6qRYUz3MRp4tHO4kxKXx+oQnUiFnR7vW0
YkNtGc1RUDHwecCTFpJtPb7Yu/E=
-----END CERTIFICATE-----
`)
	differentFrontProxyCA := []byte(`-----BEGIN CERTIFICATE-----
MIIBqDCCAU2gAwIBAgIUfbqeieihh/oERbfvRm38XvS/xHAwCgYIKoZIzj0EAwIw
GjEYMBYGA1UEAxMPSW50ZXJtZWRpYXRlLUNBMCAXDTE2MTAxMTA1MDYwMFoYDzIx
MTYwOTE3MDUwNjAwWjAUMRIwEAYDVQQDEwlNeSBDbGllbnQwWTATBgcqhkjOPQIB
BggqhkjOPQMBBwNCAARv6N4R/sjMR65iMFGNLN1GC/vd7WhDW6J4X/iAjkRLLnNb
KbRG/AtOUZ+7upJ3BWIRKYbOabbQGQe2BbKFiap4o3UwczAOBgNVHQ8BAf8EBAMC
BaAwEwYDVR0lBAwwCgYIKwYBBQUHAwIwDAYDVR0TAQH/BAIwADAdBgNVHQ4EFgQU
K/pZOWpNcYai6eHFpmJEeFpeQlEwHwYDVR0jBBgwFoAUX6nQlxjfWnP6aM1meO/Q
a6b3a9kwCgYIKoZIzj0EAwIDSQAwRgIhAIWTKw/sjJITqeuNzJDAKU4xo1zL+xJ5
MnVCuBwfwDXCAiEAw/1TA+CjPq9JC5ek1ifR0FybTURjeQqYkKpve1dveps=
-----END CERTIFICATE-----

`)
	clientCAFilename := ""
	frontProxyCAFilename := ""

	_, kubeconfig := framework.StartTestServer(t, stopCh, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.GenericServerRunOptions.MaxRequestBodyBytes = 1024 * 1024
			clientCAFilename = opts.Authentication.ClientCert.ClientCA
			frontProxyCAFilename = opts.Authentication.RequestHeader.ClientCAFile
			dynamiccertificates.FileRefreshDuration = 1 * time.Second
		},
	})
	apiserverURL, err := url.Parse(kubeconfig.Host)
	if err != nil {
		t.Fatal(err)
	}

	// when we run this the second time, we know which one we are expecting
	acceptableCAs := []string{}
	tlsConfig := &tls.Config{
		InsecureSkipVerify: true,
		GetClientCertificate: func(hello *tls.CertificateRequestInfo) (*tls.Certificate, error) {
			acceptableCAs = []string{}
			for _, curr := range hello.AcceptableCAs {
				acceptableCAs = append(acceptableCAs, string(curr))
			}
			return &tls.Certificate{}, nil
		},
	}

	conn, err := tls.Dial("tcp", apiserverURL.Host, tlsConfig)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	if err := ioutil.WriteFile(clientCAFilename, differentClientCA, 0644); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(frontProxyCAFilename, differentFrontProxyCA, 0644); err != nil {
		t.Fatal(err)
	}

	time.Sleep(4 * time.Second)

	conn2, err := tls.Dial("tcp", apiserverURL.Host, tlsConfig)
	if err != nil {
		t.Fatal(err)
	}
	defer conn2.Close()

	expectedCAs := []string{"webhook-test.default.svc", "My Client"}
	if len(expectedCAs) != len(acceptableCAs) {
		t.Fatal(strings.Join(acceptableCAs, ":"))
	}
	for i := range expectedCAs {
		if !strings.Contains(acceptableCAs[i], expectedCAs[i]) {
			t.Errorf("expected %q, got %q", expectedCAs[i], acceptableCAs[i])
		}
	}
}

func TestServingCert(t *testing.T) {
	stopCh := make(chan struct{})
	defer close(stopCh)

	var serverKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA13f50PPWuR/InxLIoJjHdNSG+jVUd25CY7ZL2J023X2BAY+1
M6jkLR6C2nSFZnn58ubiB74/d1g/Fg1Twd419iR615A013f+qOoyFx3LFHxU1S6e
v22fgJ6ntK/+4QD5MwNgOwD8k1jN2WxHqNWn16IF4Tidbv8M9A35YHAdtYDYaOJC
kzjVztzRw1y6bKRakpMXxHylQyWmAKDJ2GSbRTbGtjr7Ji54WBfG43k94tO5X8K4
VGbz/uxrKe1IFMHNOlrjR438dbOXusksx9EIqDA9a42J3qjr5NKSqzCIbgBFl6qu
45V3A7cdRI/sJ2G1aqlWIXh2fAQiaFQAEBrPfwIDAQABAoIBAAZbxgWCjJ2d8H+x
QDZtC8XI18redAWqPU9P++ECkrHqmDoBkalanJEwS1BDDATAKL4gTh9IX/sXoZT3
A7e+5PzEitN9r/GD2wIFF0FTYcDTAnXgEFM52vEivXQ5lV3yd2gn+1kCaHG4typp
ZZv34iIc5+uDjjHOWQWCvA86f8XxX5EfYH+GkjfixTtN2xhWWlfi9vzYeESS4Jbt
tqfH0iEaZ1Bm/qvb8vFgKiuSTOoSpaf+ojAdtPtXDjf1bBtQQG+RSQkP59O/taLM
FCVuRrU8EtdB0+9anwmAP+O2UqjL5izA578lQtdIh13jHtGEgOcnfGNUphK11y9r
Mg5V28ECgYEA9fwI6Xy1Rb9b9irp4bU5Ec99QXa4x2bxld5cDdNOZWJQu9OnaIbg
kw/1SyUkZZCGMmibM/BiWGKWoDf8E+rn/ujGOtd70sR9U0A94XMPqEv7iHxhpZmD
rZuSz4/snYbOWCZQYXFoD/nqOwE7Atnz7yh+Jti0qxBQ9bmkb9o0QW8CgYEA4D3d
okzodg5QQ1y9L0J6jIC6YysoDedveYZMd4Un9bKlZEJev4OwiT4xXmSGBYq/7dzo
OJOvN6qgPfibr27mSB8NkAk6jL/VdJf3thWxNYmjF4E3paLJ24X31aSipN1Ta6K3
KKQUQRvixVoI1q+8WHAubBDEqvFnNYRHD+AjKvECgYBkekjhpvEcxme4DBtw+OeQ
4OJXJTmhKemwwB12AERboWc88d3GEqIVMEWQJmHRotFOMfCDrMNfOxYv5+5t7FxL
gaXHT1Hi7CQNJ4afWrKgmjjqrXPtguGIvq2fXzjVt8T9uNjIlNxe+kS1SXFjXsgH
ftDY6VgTMB0B4ozKq6UAvQKBgQDER8K5buJHe+3rmMCMHn+Qfpkndr4ftYXQ9Kn4
MFiy6sV0hdfTgRzEdOjXu9vH/BRVy3iFFVhYvIR42iTEIal2VaAUhM94Je5cmSyd
eE1eFHTqfRPNazmPaqttmSc4cfa0D4CNFVoZR6RupIl6Cect7jvkIaVUD+wMXxWo
osOFsQKBgDLwVhZWoQ13RV/jfQxS3veBUnHJwQJ7gKlL1XZ16mpfEOOVnJF7Es8j
TIIXXYhgSy/XshUbsgXQ+YGliye/rXSCTXHBXvWShOqxEMgeMYMRkcm8ZLp/DH7C
kC2pemkLPUJqgSh1PASGcJbDJIvFGUfP69tUCYpHpk3nHzexuAg3
-----END RSA PRIVATE KEY-----`)

	var serverCert = []byte(`-----BEGIN CERTIFICATE-----
MIIDQDCCAiigAwIBAgIJANWw74P5KJk2MA0GCSqGSIb3DQEBCwUAMDQxMjAwBgNV
BAMMKWdlbmVyaWNfd2ViaG9va19hZG1pc3Npb25fcGx1Z2luX3Rlc3RzX2NhMCAX
DTE3MTExNjAwMDUzOVoYDzIyOTEwOTAxMDAwNTM5WjAjMSEwHwYDVQQDExh3ZWJo
b29rLXRlc3QuZGVmYXVsdC5zdmMwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEK
AoIBAQDXd/nQ89a5H8ifEsigmMd01Ib6NVR3bkJjtkvYnTbdfYEBj7UzqOQtHoLa
dIVmefny5uIHvj93WD8WDVPB3jX2JHrXkDTXd/6o6jIXHcsUfFTVLp6/bZ+Anqe0
r/7hAPkzA2A7APyTWM3ZbEeo1afXogXhOJ1u/wz0DflgcB21gNho4kKTONXO3NHD
XLpspFqSkxfEfKVDJaYAoMnYZJtFNsa2OvsmLnhYF8bjeT3i07lfwrhUZvP+7Gsp
7UgUwc06WuNHjfx1s5e6ySzH0QioMD1rjYneqOvk0pKrMIhuAEWXqq7jlXcDtx1E
j+wnYbVqqVYheHZ8BCJoVAAQGs9/AgMBAAGjZDBiMAkGA1UdEwQCMAAwCwYDVR0P
BAQDAgXgMB0GA1UdJQQWMBQGCCsGAQUFBwMCBggrBgEFBQcDATApBgNVHREEIjAg
hwR/AAABghh3ZWJob29rLXRlc3QuZGVmYXVsdC5zdmMwDQYJKoZIhvcNAQELBQAD
ggEBAD/GKSPNyQuAOw/jsYZesb+RMedbkzs18sSwlxAJQMUrrXwlVdHrA8q5WhE6
ABLqU1b8lQ8AWun07R8k5tqTmNvCARrAPRUqls/ryER+3Y9YEcxEaTc3jKNZFLbc
T6YtcnkdhxsiO136wtiuatpYL91RgCmuSpR8+7jEHhuFU01iaASu7ypFrUzrKHTF
bKwiLRQi1cMzVcLErq5CDEKiKhUkoDucyARFszrGt9vNIl/YCcBOkcNvM3c05Hn3
M++C29JwS3Hwbubg6WO3wjFjoEhpCwU6qRYUz3MRp4tHO4kxKXx+oQnUiFnR7vW0
YkNtGc1RUDHwecCTFpJtPb7Yu/E=
-----END CERTIFICATE-----`)

	var servingCertPath string

	_, kubeconfig := framework.StartTestServer(t, stopCh, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.GenericServerRunOptions.MaxRequestBodyBytes = 1024 * 1024
			servingCertPath = opts.SecureServing.ServerCert.CertDirectory
			dynamiccertificates.FileRefreshDuration = 1 * time.Second
		},
	})
	apiserverURL, err := url.Parse(kubeconfig.Host)
	if err != nil {
		t.Fatal(err)
	}

	// when we run this the second time, we know which one we are expecting
	acceptableCerts := [][]byte{}
	tlsConfig := &tls.Config{
		InsecureSkipVerify: true,
		VerifyPeerCertificate: func(rawCerts [][]byte, verifiedChains [][]*x509.Certificate) error {
			acceptableCerts = make([][]byte, 0, len(rawCerts))
			for _, r := range rawCerts {
				acceptableCerts = append(acceptableCerts, r)
			}
			return nil
		},
	}

	conn, err := tls.Dial("tcp", apiserverURL.Host, tlsConfig)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	if err := ioutil.WriteFile(path.Join(servingCertPath, "apiserver.key"), serverKey, 0644); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(path.Join(servingCertPath, "apiserver.crt"), serverCert, 0644); err != nil {
		t.Fatal(err)
	}

	time.Sleep(4 * time.Second)

	conn2, err := tls.Dial("tcp", apiserverURL.Host, tlsConfig)
	if err != nil {
		t.Fatal(err)
	}
	defer conn2.Close()

	cert, err := tls.X509KeyPair(serverCert, serverKey)
	if err != nil {
		t.Fatal(err)
	}

	expectedCerts := cert.Certificate
	if len(expectedCerts) != len(acceptableCerts) {
		var certs []string
		for _, a := range acceptableCerts {
			certs = append(certs, base64.StdEncoding.EncodeToString(a))
		}
		t.Fatalf("Unexpected number of certs: %v", strings.Join(certs, ":"))
	}
	for i := range expectedCerts {
		if !bytes.Equal(acceptableCerts[i], expectedCerts[i]) {
			t.Errorf("expected %q, got %q", base64.StdEncoding.EncodeToString(expectedCerts[i]), base64.StdEncoding.EncodeToString(acceptableCerts[i]))
		}
	}
}
