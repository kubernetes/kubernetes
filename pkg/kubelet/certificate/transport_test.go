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

package certificate

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"math/big"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	certificatesclient "k8s.io/client-go/kubernetes/typed/certificates/v1beta1"
	"k8s.io/client-go/rest"
)

var (
	client1CertData = newCertificateData(`-----BEGIN CERTIFICATE-----
MIICBDCCAW2gAwIBAgIJAPgVBh+4xbGoMA0GCSqGSIb3DQEBCwUAMBsxGTAXBgNV
BAMMEHdlYmhvb2tfdGVzdHNfY2EwIBcNMTcwNzI4MjMxNTI4WhgPMjI5MTA1MTMy
MzE1MjhaMB8xHTAbBgNVBAMMFHdlYmhvb2tfdGVzdHNfY2xpZW50MIGfMA0GCSqG
SIb3DQEBAQUAA4GNADCBiQKBgQDkGXXSm6Yun5o3Jlmx45rItcQ2pmnoDk4eZfl0
rmPa674s2pfYo3KywkXQ1Fp3BC8GUgzPLSfJ8xXya9Lg1Wo8sHrDln0iRg5HXxGu
uFNhRBvj2S0sIff0ZG/IatB9I6WXVOUYuQj6+A0CdULNj1vBqH9+7uWbLZ6lrD4b
a44x/wIDAQABo0owSDAJBgNVHRMEAjAAMAsGA1UdDwQEAwIF4DAdBgNVHSUEFjAU
BggrBgEFBQcDAgYIKwYBBQUHAwEwDwYDVR0RBAgwBocEfwAAATANBgkqhkiG9w0B
AQsFAAOBgQCpN27uh/LjUVCaBK7Noko25iih/JSSoWzlvc8CaipvSPofNWyGx3Vu
OdcSwNGYX/pp4ZoAzFij/Y5u0vKTVLkWXATeTMVmlPvhmpYjj9gPkCSY6j/SiKlY
kGy0xr+0M5UQkMBcfIh9oAp9um1fZHVWAJAGP/ikZgkcUey0LmBn8w==
-----END CERTIFICATE-----`, `-----BEGIN RSA PRIVATE KEY-----
MIICWwIBAAKBgQDkGXXSm6Yun5o3Jlmx45rItcQ2pmnoDk4eZfl0rmPa674s2pfY
o3KywkXQ1Fp3BC8GUgzPLSfJ8xXya9Lg1Wo8sHrDln0iRg5HXxGuuFNhRBvj2S0s
Iff0ZG/IatB9I6WXVOUYuQj6+A0CdULNj1vBqH9+7uWbLZ6lrD4ba44x/wIDAQAB
AoGAZbWwowvCq1GBq4vPPRI3h739Uz0bRl1ymf1woYXNguXRtCB4yyH+2BTmmrrF
6AIWkePuUEdbUaKyK5nGu3iOWM+/i6NP3kopQANtbAYJ2ray3kwvFlhqyn1bxX4n
gl/Cbdw1If4zrDrB66y8mYDsjzK7n/gFaDNcY4GArjvOXKkCQQD9Lgv+WD73y4RP
yS+cRarlEeLLWVsX/pg2oEBLM50jsdUnrLSW071MjBgP37oOXzqynF9SoDbP2Y5C
x+aGux9LAkEA5qPlQPv0cv8Wc3qTI+LixZ/86PPHKWnOnwaHm3b9vQjZAkuVQg3n
Wgg9YDmPM87t3UFH7ZbDihUreUxwr9ZjnQJAZ9Z95shMsxbOYmbSVxafu6m1Sc+R
M+sghK7/D5jQpzYlhUspGf8n0YBX0hLhXUmjamQGGH5LXL4Owcb4/mM6twJAEVio
SF/qva9jv+GrKVrKFXT374lOJFY53Qn/rvifEtWUhLCslCA5kzLlctRBafMZPrfH
Mh5RrJP1BhVysDbenQJASGcc+DiF7rB6K++ZGyC11E2AP29DcZ0pgPESSV7npOGg
+NqPRZNVCSZOiVmNuejZqmwKhZNGZnBFx1Y+ChAAgw==
-----END RSA PRIVATE KEY-----`)
	client2CertData = newCertificateData(`-----BEGIN CERTIFICATE-----
MIICBDCCAW2gAwIBAgIJAPgVBh+4xbGnMA0GCSqGSIb3DQEBCwUAMBsxGTAXBgNV
BAMMEHdlYmhvb2tfdGVzdHNfY2EwIBcNMTcwNzI4MjMxNTI4WhgPMjI5MTA1MTMy
MzE1MjhaMB8xHTAbBgNVBAMMFHdlYmhvb2tfdGVzdHNfY2xpZW50MIGfMA0GCSqG
SIb3DQEBAQUAA4GNADCBiQKBgQDQQLzbrmHbtlxE7wViaoXFp5tQx7zzM2Ed7O1E
gs3JUws5KkPbNrejLwixvLkzzU152M43UGsyKDn7HPyjXDogTZSW6C257XpYodk3
S/gZS9oZtPss4UJuJioQk/M8X1ZjYP8kCTArOvVRJeNQL8GM7h5QQ6J5LUq+IdZb
T0retQIDAQABo0owSDAJBgNVHRMEAjAAMAsGA1UdDwQEAwIF4DAdBgNVHSUEFjAU
BggrBgEFBQcDAgYIKwYBBQUHAwEwDwYDVR0RBAgwBocEfwAAATANBgkqhkiG9w0B
AQsFAAOBgQBdAxoU5YAmp0d+5b4qg/xOGC5rKcnksQEXYoGwFBWwaKvh9oUlGGxI
A5Ykf2TEl24br4tLmicpdxUX4H4PbkdPxOjM9ghIKlmgHo8vBRC0iVIwYgQsw1W8
ETY34Or+PJqaeslqx/t7kUKY5UIF9DLVolsIiAHveJNR2uBWiP0KiQ==
-----END CERTIFICATE-----`, `-----BEGIN RSA PRIVATE KEY-----
MIICXQIBAAKBgQDQQLzbrmHbtlxE7wViaoXFp5tQx7zzM2Ed7O1Egs3JUws5KkPb
NrejLwixvLkzzU152M43UGsyKDn7HPyjXDogTZSW6C257XpYodk3S/gZS9oZtPss
4UJuJioQk/M8X1ZjYP8kCTArOvVRJeNQL8GM7h5QQ6J5LUq+IdZbT0retQIDAQAB
AoGBAMFjTL4IKvG4X+jXub1RxFXvNkkGos2Jaec7TH5xpZ4OUv7L4+We41tTYxSC
d83GGetLzPwK3vDd8DHkEiu1incket78rwmQ89LnQNyM0B5ejaTjW2zHcvKJ0Mtn
nM32juQfq8St9JZVweS87k8RkLt9cOrg6219MRbFO+1Vn8WhAkEA+/rqHCspBdXr
7RL+H63k7RjqBllVEYlw1ukqTw1gp5IImmeOwgl3aRrJJfFV6gxxEqQ4CCb2vf9M
yjrGEvP9KQJBANOTPcpskT/0dyipsAkvLFZTKjN+4fdfq37H3dVgMR6oQcMJwukd
cEio1Hx+XzXuD0RHXighq7bUzel+IqzRuq0CQBJkzpIf1G7InuA/cq19VCi6mNq9
yqftEH+fpab/ov6YemhLBvDDICRcADL02wCqx9ZEhpKRxZE5AbIBeFQJ24ECQG4f
9cmnOPNRC7TengIpy6ojH5QuNu/LnDghUBYAO5D5g0FBk3JDIG6xceha3rPzdX7U
pu28mORRX9xpCyNpBwECQQCtDNZoehdPVuZA3Wocno31Rjmuy83ajgRRuEzqv0tj
uC6Jo2eLcSV1sSdzTjaaWdM6XeYj6yHOAm8ZBIQs7m6V
-----END RSA PRIVATE KEY-----`)
)

type certificateData struct {
	keyPEM         []byte
	certificatePEM []byte
	certificate    *tls.Certificate
}

func newCertificateData(certificatePEM string, keyPEM string) *certificateData {
	certificate, err := tls.X509KeyPair([]byte(certificatePEM), []byte(keyPEM))
	if err != nil {
		panic(fmt.Sprintf("Unable to initialize certificate: %v", err))
	}
	certs, err := x509.ParseCertificates(certificate.Certificate[0])
	if err != nil {
		panic(fmt.Sprintf("Unable to initialize certificate leaf: %v", err))
	}
	certificate.Leaf = certs[0]
	return &certificateData{
		keyPEM:         []byte(keyPEM),
		certificatePEM: []byte(certificatePEM),
		certificate:    &certificate,
	}
}

type fakeManager struct {
	cert    atomic.Value // Always a *tls.Certificate
	healthy bool
}

func (f *fakeManager) SetCertificateSigningRequestClient(certificatesclient.CertificateSigningRequestInterface) error {
	return nil
}

func (f *fakeManager) ServerHealthy() bool { return f.healthy }

func (f *fakeManager) Start() {}

func (f *fakeManager) Current() *tls.Certificate {
	if val := f.cert.Load(); val != nil {
		return val.(*tls.Certificate)
	}
	return nil
}

func (f *fakeManager) setCurrent(cert *tls.Certificate) {
	f.cert.Store(cert)
}

func TestRotateShutsDownConnections(t *testing.T) {

	// This test fails if you comment out the t.closeAllConns() call in
	// transport.go and don't close connections on a rotate.

	stop := make(chan struct{})
	defer close(stop)

	m := new(fakeManager)
	m.setCurrent(client1CertData.certificate)

	// The last certificate we've seen.
	lastSeenLeafCert := new(atomic.Value) // Always *x509.Certificate

	lastSerialNumber := func() *big.Int {
		if cert := lastSeenLeafCert.Load(); cert != nil {
			return cert.(*x509.Certificate).SerialNumber
		}
		return big.NewInt(0)
	}

	h := func(w http.ResponseWriter, r *http.Request) {
		if r.TLS != nil && len(r.TLS.PeerCertificates) != 0 {
			// Record the last TLS certificate the client sent.
			lastSeenLeafCert.Store(r.TLS.PeerCertificates[0])
		}
		w.Write([]byte(`{}`))
	}

	s := httptest.NewUnstartedServer(http.HandlerFunc(h))
	s.TLS = &tls.Config{
		// Just request a cert, we don't need to verify it.
		ClientAuth: tls.RequestClientCert,
	}
	s.StartTLS()
	defer s.Close()

	c := &rest.Config{
		Host: s.URL,
		TLSClientConfig: rest.TLSClientConfig{
			// We don't care about the server's cert.
			Insecure: true,
		},
		ContentConfig: rest.ContentConfig{
			// This is a hack. We don't actually care about the serializer.
			NegotiatedSerializer: serializer.NegotiatedSerializerWrapper(runtime.SerializerInfo{}),
		},
	}

	// Check for a new cert every 10 milliseconds
	if _, err := updateTransport(stop, 10*time.Millisecond, c, m, 0); err != nil {
		t.Fatal(err)
	}

	client, err := rest.UnversionedRESTClientFor(c)
	if err != nil {
		t.Fatal(err)
	}

	if err := client.Get().Do().Error(); err != nil {
		t.Fatal(err)
	}
	firstCertSerial := lastSerialNumber()

	// Change the manager's certificate. This should cause the client to shut down
	// its connections to the server.
	m.setCurrent(client2CertData.certificate)

	for i := 0; i < 5; i++ {
		time.Sleep(time.Millisecond * 10)
		client.Get().Do()
		if firstCertSerial.Cmp(lastSerialNumber()) != 0 {
			// The certificate changed!
			return
		}
	}

	t.Errorf("certificate rotated but client never reconnected with new cert")
}
