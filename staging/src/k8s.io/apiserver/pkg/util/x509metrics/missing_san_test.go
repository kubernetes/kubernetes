/*
Copyright 2020 The Kubernetes Authors.

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

package x509metrics

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
)

// taken from pkg/util/webhook/certs_test.go
var caCert = []byte(`-----BEGIN CERTIFICATE-----
MIIDGTCCAgGgAwIBAgIUOS2MkobR2t4rguefcC78gLuXkc0wDQYJKoZIhvcNAQEL
BQAwGzEZMBcGA1UEAwwQd2ViaG9va190ZXN0c19jYTAgFw0yMDEwMDcxMjMxNDFa
GA8yMjk0MDcyMzEyMzE0MVowGzEZMBcGA1UEAwwQd2ViaG9va190ZXN0c19jYTCC
ASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAMy8o1sRSe8viuSxh8a7Ou8n
vyDJxeJA9son+clZdSwa1YpbhSsYPr2svTGLT4ZxX3JkvHZGDaB+G6QrZNIWOssJ
4+fjvEsVSnk9q7Yr8wX8QdShksnSP0qdB1xFKlqwTFcZyQHqjlyctV93/LEosM9D
RZzgPpXJuC6wD7qdLRSPxQwuegZYcMGT2Y4/8CfBgiEUkZNrGYJiDFrblI0N9jX2
jUeP/g8xWJIKLTBedVbAzj02Y66WdNecYcyTMROdclPB1IbF4ABbCGynnP0fbiaf
0+4v0pLedqwCYSWD/ujajyDtYuhI8U+OkENwkZ/h5jw/6aWq7esMZDbEAT6UPgEC
AwEAAaNTMFEwHQYDVR0OBBYEFBgvyZWkRJCRjxclYA0mMlnzc/GmMB8GA1UdIwQY
MBaAFBgvyZWkRJCRjxclYA0mMlnzc/GmMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZI
hvcNAQELBQADggEBAHigaFJ8DqYFMGXNUT5lYM8oLLWYcbivGxR7ofDm8rXWGLdQ
tKsWeusSlgpzeO1PKCzwSrYQhlFIZI6AH+ch7EAWt84MfAaMz/1DiXF6Q8fAcR0h
QoyFIAKTiVkcgOQjSQOIM2SS5cyGeDRaHGVWfaJOwdIYo6ctFzI2asPJ4yU0QsA7
0WTD2+sBG6AXGhfafGUHEmou8sGQ+QT8rgi4hs1bfyHuT5dPgC4TbwknD1G8pMqm
ID3CIiCF8hhF5C2hjrW0LTJ6zTlg1c1K0NmmUL1ucsfzEk//c7GsU8dk+FYmtW9A
VzryJj1NmnSqA3zd3jBMuK37Ei3pRvVbO7Uxf14=
-----END CERTIFICATE-----`)

var serverKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA0Gi/3oXPU0oRP38xNAQ1js4Py1fVqy5OW+3rcaZONdUiiHJ4
6K1mIMKuEH2nrFAq2cFiXIm9prrSN8hyLY9lUguyAlecdHe6he/7/PAoX7NM8NX1
LzOfUotcO+iZtrnMVCut2vlqjsGsnxppLKnFnp17ycOy7yKCuQHowPln5xp6JkhR
F0bxDXIFFmve4uHHlRsWcEoNQRJZTaOMrfrFYfqbc5iBitI7/5xgMPZNq282TQDc
GeJmbFHEu/L1qO6uTg1LkbNbZKNfXeAU46K+eecGCwPyTMlDYuREdoVpH9GYW7Cx
OIRTYgCLWVok96yk2jeVCq21BbxZ8+NBJ2rhqQIDAQABAoIBAB8bY3gdVOLDrWti
2r8+2ZelHiplw9i3Iq8KBhiCkC3s0Ci5nV5tc070f/KqLrrDhIHYIYxaatpWDEaT
PqeaPa9PW5SJ6ypfLJINTfllB0Gxi4xvAxe2htNVRcETaM4jUWJG2r5SeBsywUdG
M+ictoiETRPCiBS1e/mNVWZoU5/kyMApP+5U9+THKhV4V2Pi2OHon/AeTP1Yc/1A
lTB9giQIysDK11j9zbpL2ICW9HSbbeLJlsw8/wCOdnPHFn5kun6EuesOF6MingvM
vL9ZHsh6N7oOHupqRvDPImM3QTYSuBe1hTJSvhi7hmnl9dolsBKKxJz0wjCO1N+V
wdPzrwECgYEA9PH+5RSWJtUb2riDpM0QB/Cci6UEFeTJbaCSluog/EH1/E+mOo2J
VbLCMhWwF8kJYToG0sTFGbJ1J1SiaYan7pxH2YIZkgXePC8Vj5WJnLhv05kKRanq
kOE1hVnaaCeFJFiLjDW2WeEaNLo6Ap1Qnb6ObzwV+0QvWCMUnVQwfjECgYEA2dCh
JKDXdsuLUklTc39VKCXXhqO/5FFOzNrZhPEo6KV2mY0BZnNjau4Ff2F6UJb6NMza
fFSLScEZTdbCv5lElGAxJueqC+p2LR3dS/1JfdWJ0zrP1BZa+MWr8O510Go/NOC2
/s5cR2aVBdJ2WK4d/3XShOr6W8T6hPisr5wFZPkCgYBptUIWpNLEAXZa5wRRC/pe
ItW8YkOoGytet0xr+rCvjNvWvpzzaf+Zz2KFcNyk9yqoHf2x2h9hnqV2iszok6dH
j4RmdwIIBaZJ/NvmMlfIHcSM4eAP/mtviPGrEgLyrhOEgv3+TXPbyAyiMrg0RqXy
3bjkgl7OKDfyZnlQCHRBEQKBgCfmLK6N/AoZzQKcxfmhOJMrI2jZdBw5vKqP6EqO
9oRvUuNbzgbbWjnLMhycWZCLp3emkts1jXJMOftlPLVmOQbI/Bf5Vc/q+gzXrKLv
2deAF0gnPMzH75AkfZObyt8Lp1pjU4IngQXfR6sSW3VxJ7OU/KQ2evf2hEF5YACn
HuHZAoGBAI+i6KI0WiWadenDctkuzLgxEPXaQ3oLBJjhE9CjpcdF6mRMWaU8lPgj
D1bo9L8wxvqIW5qIrX9dKnGgYAxnomhBNQn3C+5XDgtq6UiANalilqs3AoaWlQiF
WKaPuWf2T2ypFVzdzPl/0fsFUo8Rw5D4VO4nHeglOrkGQx+OdXA6
-----END RSA PRIVATE KEY-----`)

var serverCert = []byte(`-----BEGIN CERTIFICATE-----
MIIDHzCCAgegAwIBAgIUZIt+6zrmR41Be/CrcPHLsj3pCAQwDQYJKoZIhvcNAQEL
BQAwGzEZMBcGA1UEAwwQd2ViaG9va190ZXN0c19jYTAgFw0yMDEwMDcxMjMxNDFa
GA8yMjk0MDcyMzEyMzE0MVowHzEdMBsGA1UEAwwUd2ViaG9va190ZXN0c19zZXJ2
ZXIwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDQaL/ehc9TShE/fzE0
BDWOzg/LV9WrLk5b7etxpk411SKIcnjorWYgwq4QfaesUCrZwWJcib2mutI3yHIt
j2VSC7ICV5x0d7qF7/v88Chfs0zw1fUvM59Si1w76Jm2ucxUK63a+WqOwayfGmks
qcWenXvJw7LvIoK5AejA+WfnGnomSFEXRvENcgUWa97i4ceVGxZwSg1BEllNo4yt
+sVh+ptzmIGK0jv/nGAw9k2rbzZNANwZ4mZsUcS78vWo7q5ODUuRs1tko19d4BTj
or555wYLA/JMyUNi5ER2hWkf0ZhbsLE4hFNiAItZWiT3rKTaN5UKrbUFvFnz40En
auGpAgMBAAGjVTBTMAkGA1UdEwQCMAAwCwYDVR0PBAQDAgXgMB0GA1UdJQQWMBQG
CCsGAQUFBwMCBggrBgEFBQcDATAaBgNVHREEEzARhwR/AAABgglsb2NhbGhvc3Qw
DQYJKoZIhvcNAQELBQADggEBAFXNoW48IHJAcO84Smb+/k8DvJBwOorOPspJ/6DY
pYITCANq9kQ54bv3LgPuel5s2PytVL1dVQmAya1cT9kG3nIjNaulR2j5Sgt0Ilyd
Dk/HOE/zBi6KyifV3dgQSbzua8AI9VboR3o3FhmA9C9jDDxAS+q9+NQjB40/aG8m
TBx+oKgeYHee5llKNTsY1Jqh6TT47om70+sjvmgZ4blAV7ft+WG/h3ZVtAZJuFee
tchgUEpGR8ZGyK0r/vWBIKHNSqtG5gdOS9swQLdUFG90OivhddKUU8Zt52uUXbc/
/ggEd4dM4X6B21xKJQY6vCnTvHFXcVJezV3g1xaNN0yR0DA=
-----END CERTIFICATE-----`)

var serverCertNoSAN = []byte(`-----BEGIN CERTIFICATE-----
MIIC+DCCAeCgAwIBAgIUZIt+6zrmR41Be/CrcPHLsj3pCAUwDQYJKoZIhvcNAQEL
BQAwGzEZMBcGA1UEAwwQd2ViaG9va190ZXN0c19jYTAgFw0yMDEwMDcxMjMxNDFa
GA8yMjk0MDcyMzEyMzE0MVowFDESMBAGA1UEAwwJbG9jYWxob3N0MIIBIjANBgkq
hkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA0Gi/3oXPU0oRP38xNAQ1js4Py1fVqy5O
W+3rcaZONdUiiHJ46K1mIMKuEH2nrFAq2cFiXIm9prrSN8hyLY9lUguyAlecdHe6
he/7/PAoX7NM8NX1LzOfUotcO+iZtrnMVCut2vlqjsGsnxppLKnFnp17ycOy7yKC
uQHowPln5xp6JkhRF0bxDXIFFmve4uHHlRsWcEoNQRJZTaOMrfrFYfqbc5iBitI7
/5xgMPZNq282TQDcGeJmbFHEu/L1qO6uTg1LkbNbZKNfXeAU46K+eecGCwPyTMlD
YuREdoVpH9GYW7CxOIRTYgCLWVok96yk2jeVCq21BbxZ8+NBJ2rhqQIDAQABozkw
NzAJBgNVHRMEAjAAMAsGA1UdDwQEAwIF4DAdBgNVHSUEFjAUBggrBgEFBQcDAgYI
KwYBBQUHAwEwDQYJKoZIhvcNAQELBQADggEBAGgrpuQ4n0W/TaaXhdbfFELziXoN
eT89eFYOqgtx/o97sj8B/y+n8tm+lBopfmdnifta3e8iNVULAd6JKjBhkFL1NY3Q
tR+VqT8uEZthDM69cyuGTv1GybnUCY9mtW9dSgHHkcJNsZGn9oMTCYDcBrgoA4s3
vZc4wuPj8wyiSJBDYMNZfLWHCvLTOa0XUfh0Pf0/d26KuAUTNQkJZLZ5cbNXZuu3
fVN5brtOw+Md5nUa60z+Xp0ESaGvOLUjnmd2SWUfAzcbFbbV4fAyYZF/93zCVTJ1
ig9gRWmPqLcDDLf9LPyDR5yHRTSF4USH2ykun4PiPfstjfv0xwddWgG2+S8=
-----END CERTIFICATE-----`)

// Test_checkForHostnameError tests that the upstream message for remote server
// certificate's hostname hasn't changed when no SAN extension is present and that
// the metrics counter increases properly when such an error is encountered
//
// Requires GODEBUG=x509ignoreCN=0 to not be set in the environment
func TestCheckForHostnameError(t *testing.T) {
	tests := []struct {
		name            string
		serverCert      []byte
		counterIncrease bool
	}{
		{
			name:            "no SAN",
			serverCert:      serverCertNoSAN,
			counterIncrease: true,
		},
		{
			name:       "with SAN",
			serverCert: serverCert,
		},
	}

	// register the test metrics
	x509MissingSANCounter := metrics.NewCounter(&metrics.CounterOpts{Name: "Test_checkForHostnameError"})
	registry := testutil.NewFakeKubeRegistry("0.0.0")
	registry.MustRegister(x509MissingSANCounter)

	var lastCounterVal int
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tlsServer, serverURL := testServer(t, tt.serverCert)
			defer tlsServer.Close()

			client := tlsServer.Client()
			req, err := http.NewRequest(http.MethodGet, serverURL.String(), nil)
			if err != nil {
				t.Fatalf("failed to create an http request: %v", err)
			}

			_, err = client.Transport.RoundTrip(req)

			checkForHostnameError(err, x509MissingSANCounter)

			errorCounterVal := getSingleCounterValueFromRegistry(t, registry, "Test_checkForHostnameError")
			if errorCounterVal == -1 {
				t.Fatalf("failed to get the error counter from the registry")
			}

			if tt.counterIncrease && errorCounterVal != lastCounterVal+1 {
				t.Errorf("expected the Test_checkForHostnameError metrics to increase by 1 from %d, but it is %d", lastCounterVal, errorCounterVal)
			}

			if !tt.counterIncrease && errorCounterVal != lastCounterVal {
				t.Errorf("expected the Test_checkForHostnameError metrics to stay the same (%d), but it is %d", lastCounterVal, errorCounterVal)
			}

			lastCounterVal = errorCounterVal
		})
	}
}

func TestCheckRespForNoSAN(t *testing.T) {
	tests := []struct {
		name            string
		serverCert      []byte
		counterIncrease bool
	}{
		{
			name: "no certs",
		},
		{
			name:            "no SAN",
			serverCert:      serverCertNoSAN,
			counterIncrease: true,
		},
		{
			name:       "with SAN",
			serverCert: serverCert,
		},
	}

	// register the test metrics
	x509MissingSANCounter := metrics.NewCounter(&metrics.CounterOpts{Name: "Test_checkRespForNoSAN"})
	registry := testutil.NewFakeKubeRegistry("0.0.0")
	registry.MustRegister(x509MissingSANCounter)

	var lastCounterVal int
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var tlsConnectionState *tls.ConnectionState
			if tt.serverCert != nil {
				block, _ := pem.Decode([]byte(tt.serverCert))
				if block == nil {
					t.Fatal("failed to parse certificate PEM")
				}

				serverCert, err := x509.ParseCertificate(block.Bytes)
				if err != nil {
					t.Fatalf("failed to parse certificate: %v", err)
				}

				tlsConnectionState = &tls.ConnectionState{
					PeerCertificates: []*x509.Certificate{serverCert},
				}
			}

			resp := &http.Response{
				TLS: tlsConnectionState,
			}

			checkRespForNoSAN(resp, x509MissingSANCounter)

			errorCounterVal := getSingleCounterValueFromRegistry(t, registry, "Test_checkRespForNoSAN")
			if errorCounterVal == -1 {
				t.Fatalf("failed to get the error counter from the registry")
			}

			if tt.counterIncrease && errorCounterVal != lastCounterVal+1 {
				t.Errorf("expected the Test_checkRespForNoSAN metrics to increase by 1 from %d, but it is %d", lastCounterVal, errorCounterVal)
			}

			if !tt.counterIncrease && errorCounterVal != lastCounterVal {
				t.Errorf("expected the Test_checkRespForNoSAN metrics to stay the same (%d), but it is %d", lastCounterVal, errorCounterVal)
			}

			lastCounterVal = errorCounterVal
		})
	}
}

func testServer(t *testing.T, serverCert []byte) (*httptest.Server, *url.URL) {
	rootCAs := x509.NewCertPool()
	rootCAs.AppendCertsFromPEM(caCert)

	cert, err := tls.X509KeyPair(serverCert, serverKey)
	if err != nil {
		t.Fatalf("failed to init x509 cert/key pair: %v", err)
	}
	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{cert},
		RootCAs:      rootCAs,
	}

	tlsServer := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.WriteHeader(200)
		w.Write([]byte("ok"))
	}))

	tlsServer.TLS = tlsConfig
	tlsServer.StartTLS()

	serverURL, err := url.Parse(tlsServer.URL)
	if err != nil {
		tlsServer.Close()
		t.Fatalf("failed to parse the testserver URL: %v", err)
	}
	serverURL.Host = net.JoinHostPort("localhost", serverURL.Port())

	return tlsServer, serverURL
}

func getSingleCounterValueFromRegistry(t *testing.T, r metrics.Gatherer, name string) int {
	mfs, err := r.Gather()
	if err != nil {
		t.Logf("failed to gather local registry metrics: %v", err)
		return -1
	}

	for _, mf := range mfs {
		if mf.Name != nil && *mf.Name == name {
			mfMetric := mf.GetMetric()
			for _, m := range mfMetric {
				if m.GetCounter() != nil {
					return int(m.GetCounter().GetValue())
				}
			}
		}
	}

	return -1
}
