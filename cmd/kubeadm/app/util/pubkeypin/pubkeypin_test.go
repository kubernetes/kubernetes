/*
Copyright 2017 The Kubernetes Authors.

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

package pubkeypin

import (
	"crypto/x509"
	"encoding/pem"
	"strings"
	"testing"
)

// testCertPEM is a simple self-signed test certificate issued with the openssl CLI:
// openssl req -new -newkey rsa:2048 -days 36500 -nodes -x509 -keyout /dev/null -out test.crt
const testCertPEM = `
-----BEGIN CERTIFICATE-----
MIIDRDCCAiygAwIBAgIJAJgVaCXvC6HkMA0GCSqGSIb3DQEBBQUAMB8xHTAbBgNV
BAMTFGt1YmVhZG0ta2V5cGlucy10ZXN0MCAXDTE3MDcwNTE3NDMxMFoYDzIxMTcw
NjExMTc0MzEwWjAfMR0wGwYDVQQDExRrdWJlYWRtLWtleXBpbnMtdGVzdDCCASIw
DQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAK0ba8mHU9UtYlzM1Own2Fk/XGjR
J4uJQvSeGLtz1hID1IA0dLwruvgLCPadXEOw/f/IWIWcmT+ZmvIHZKa/woq2iHi5
+HLhXs7aG4tjKGLYhag1hLjBI7icqV7ovkjdGAt9pWkxEzhIYClFMXDjKpMSynu+
YX6nZ9tic1cOkHmx2yiZdMkuriRQnpTOa7bb03OC1VfGl7gHlOAIYaj4539WCOr8
+ACTUMJUFEHcRZ2o8a/v6F9GMK+7SC8SJUI+GuroXqlMAdhEv4lX5Co52enYaClN
+D9FJLRpBv2YfiCQdJRaiTvCBSxEFz6BN+PtP5l2Hs703ZWEkOqCByM6HV8CAwEA
AaOBgDB+MB0GA1UdDgQWBBRQgUX8MhK2rWBWQiPHWcKzoWDH5DBPBgNVHSMESDBG
gBRQgUX8MhK2rWBWQiPHWcKzoWDH5KEjpCEwHzEdMBsGA1UEAxMUa3ViZWFkbS1r
ZXlwaW5zLXRlc3SCCQCYFWgl7wuh5DAMBgNVHRMEBTADAQH/MA0GCSqGSIb3DQEB
BQUAA4IBAQCaAUif7Pfx3X0F08cxhx8/Hdx4jcJw6MCq6iq6rsXM32ge43t8OHKC
pJW08dk58a3O1YQSMMvD6GJDAiAfXzfwcwY6j258b1ZlI9Ag0VokvhMl/XfdCsdh
AWImnL1t4hvU5jLaImUUMlYxMcSfHBGAm7WJIZ2LdEfg6YWfZh+WGbg1W7uxLxk6
y4h5rWdNnzBHWAGf7zJ0oEDV6W6RSwNXtC0JNnLaeIUm/6xdSddJlQPwUv8YH4jX
c1vuFqTnJBPcb7W//R/GI2Paicm1cmns9NLnPR35exHxFTy+D1yxmGokpoPMdife
aH+sfuxT8xeTPb3kjzF9eJTlnEquUDLM
-----END CERTIFICATE-----`

// expectedHash can be verified using the openssl CLI.
const expectedHash = `sha256:345959acb2c3b2feb87d281961c893f62a314207ef02599f1cc4a5fb255480b3`

// testCert2PEM is a second test cert generated the same way as testCertPEM
const testCert2PEM = `
-----BEGIN CERTIFICATE-----
MIID9jCCAt6gAwIBAgIJAN5MXZDic7qYMA0GCSqGSIb3DQEBBQUAMFkxCzAJBgNV
BAYTAkFVMRMwEQYDVQQIEwpTb21lLVN0YXRlMSEwHwYDVQQKExhJbnRlcm5ldCBX
aWRnaXRzIFB0eSBMdGQxEjAQBgNVBAMTCXRlc3RDZXJ0MjAgFw0xNzA3MjQxNjA0
MDFaGA8yMTE3MDYzMDE2MDQwMVowWTELMAkGA1UEBhMCQVUxEzARBgNVBAgTClNv
bWUtU3RhdGUxITAfBgNVBAoTGEludGVybmV0IFdpZGdpdHMgUHR5IEx0ZDESMBAG
A1UEAxMJdGVzdENlcnQyMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA
0brwpJYN2ytPWzRBtZSVc3dhkQlA59AzxzqeLLkano0Pxo9NIc3T/y58nnRI8uaS
I1P7BzUfJTiUEvmAtX8NggqKK4ld/gPrU+IRww1CUYS4KCkA/0d0ctPy0JwBCjD+
b57G3rmNE8c+0jns6J96ZzNtqmv6N+ZlFBAXm1p4S+k0kGi5+hoQ6H7SYXjk2lG+
r/8jPQEjy/NSdw1dcCA0Nc6o+hPr32927dS6J9KOhBeXNYUNdbuDDmroM9/gN2e/
YMSA1olLeDPQ7Xvhk0PIyEDnHh83AffPCx5yM3htVRGddjIsPAVUJEL3z5leJtxe
fzyPghOhHJY0PXqznDQTcwIDAQABo4G+MIG7MB0GA1UdDgQWBBRP0IJqv/5rQ4Uf
SByl77dJeEapRDCBiwYDVR0jBIGDMIGAgBRP0IJqv/5rQ4UfSByl77dJeEapRKFd
pFswWTELMAkGA1UEBhMCQVUxEzARBgNVBAgTClNvbWUtU3RhdGUxITAfBgNVBAoT
GEludGVybmV0IFdpZGdpdHMgUHR5IEx0ZDESMBAGA1UEAxMJdGVzdENlcnQyggkA
3kxdkOJzupgwDAYDVR0TBAUwAwEB/zANBgkqhkiG9w0BAQUFAAOCAQEA0RIMHc10
wHHPMh9UflqBgDMF7gfbOL0juJfGloAOcohWWfMZBBJ0CQKMy3xRyoK3HmbW1eeb
iATjesw7t4VEAwf7mgKAd+eTfWYB952uq5qYJ2TI28mSofEq1Wz3RmrNkC1KCBs1
u+YMFGwyl6necV9zKCeiju4jeovI1GA38TvH7MgYln6vMJ+FbgOXj7XCpek7dQiY
KGaeSSH218mGNQaWRQw2Sm3W6cFdANoCJUph4w18s7gjtFpfV63s80hXRps+vEyv
jEQMEQpG8Ss7HGJLGLBw/xAmG0e//XS/o2dDonbGbvzToFByz8OGxjMhk6yV6hdd
+iyvsLAw/MYMSA==
-----END CERTIFICATE-----
`

// testCert is a small helper to get a test x509.Certificate from the PEM constants
func testCert(t *testing.T, pemString string) *x509.Certificate {
	// Decode the example certificate from a PEM file into a PEM block
	pemBlock, _ := pem.Decode([]byte(pemString))
	if pemBlock == nil {
		t.Fatal("failed to parse test certificate PEM")
		return nil
	}

	// Parse the PEM block into an x509.Certificate
	result, err := x509.ParseCertificate(pemBlock.Bytes)
	if err != nil {
		t.Fatalf("failed to parse test certificate: %v", err)
		return nil
	}
	return result
}

func TestSet(t *testing.T) {
	s := NewSet()
	if !s.Empty() {
		t.Error("expected a new set to be empty")
		return
	}
	err := s.Allow("xyz")
	if err == nil || !s.Empty() {
		t.Error("expected allowing junk to fail")
		return
	}

	err = s.Allow("0011223344")
	if err == nil || !s.Empty() {
		t.Error("expected allowing something too short to fail")
		return
	}

	err = s.Allow(expectedHash + expectedHash)
	if err == nil || !s.Empty() {
		t.Error("expected allowing something too long to fail")
		return
	}

	err = s.CheckAny([]*x509.Certificate{testCert(t, testCertPEM)})
	if err == nil {
		t.Error("expected test cert to not be allowed (yet)")
		return
	}

	err = s.Allow(strings.ToUpper(expectedHash))
	if err != nil || s.Empty() {
		t.Error("expected allowing uppercase expectedHash to succeed")
		return
	}

	err = s.CheckAny([]*x509.Certificate{testCert(t, testCertPEM)})
	if err != nil {
		t.Errorf("expected test cert to be allowed, but got back: %v", err)
		return
	}

	err = s.CheckAny([]*x509.Certificate{testCert(t, testCert2PEM)})
	if err == nil {
		t.Error("expected the second test cert to be disallowed")
		return
	}
}

func TestHash(t *testing.T) {
	actualHash := Hash(testCert(t, testCertPEM))
	if actualHash != expectedHash {
		t.Errorf(
			"failed to Hash() to the expected value\n\texpected: %q\n\t  actual: %q",
			expectedHash,
			actualHash,
		)
	}
}
