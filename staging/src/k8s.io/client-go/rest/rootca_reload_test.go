package rest

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"golang.org/x/net/http2"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

var firstCA = []byte(`-----BEGIN CERTIFICATE-----
MIIBqzCCAVKgAwIBAgIUOtmTzajIwqsbc6jlolHsN7kbv1swCgYIKoZIzj0EAwIw
NDELMAkGA1UEBhMCUEwxDzANBgNVBAcTBkdkYW5zazEUMBIGA1UEAxMLZXhhbXBs
ZS5uZXQwHhcNMjMxMTA3MDgyOTAwWhcNMjgxMTA1MDgyOTAwWjA0MQswCQYDVQQG
EwJQTDEPMA0GA1UEBxMGR2RhbnNrMRQwEgYDVQQDEwtleGFtcGxlLm5ldDBZMBMG
ByqGSM49AgEGCCqGSM49AwEHA0IABLpdWkE0yPXT9FO16He/0dR35ToALaTBFLqc
PLhBQd/xBsT03GK2GTbIWm5Ft7sdW44aDomecDUAFDbRYVbe9NKjQjBAMA4GA1Ud
DwEB/wQEAwIBBjAPBgNVHRMBAf8EBTADAQH/MB0GA1UdDgQWBBSwQfSZdM3VjoQw
O2FcMg2k/FXtezAKBggqhkjOPQQDAgNHADBEAiAtLJAi/isK4VzfPmm6vGMDohlp
G3qnd+dsCwlok2ZQUQIgIhEy1mCY/nYpWXJnNC/uC3YqmUgiw8qkkS3IZq00VHc=
-----END CERTIFICATE-----`)

var firstServerCert = []byte(`-----BEGIN CERTIFICATE-----
MIICCzCCAbKgAwIBAgIUXb4puzaO7YCG+RZCf4Ejf5nli60wCgYIKoZIzj0EAwIw
NDELMAkGA1UEBhMCUEwxDzANBgNVBAcTBkdkYW5zazEUMBIGA1UEAxMLZXhhbXBs
ZS5uZXQwHhcNMjMxMTA3MDgzNTAwWhcNMzMxMTA0MDgzNTAwWjA2MQswCQYDVQQG
EwJQTDEPMA0GA1UEBxMGR2RhbnNrMRYwFAYDVQQDEw1sb2NhbGhvc3QubmV0MFkw
EwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEXtlWtbQGGXqciklSB4H70SYnt8Vc7U2G
weRjKklVT3nKJuVsi91wVJ7wIgzABwdr/VWXTH3CZOxitQK7Mi09MKOBnzCBnDAO
BgNVHQ8BAf8EBAMCBaAwEwYDVR0lBAwwCgYIKwYBBQUHAwEwDAYDVR0TAQH/BAIw
ADAdBgNVHQ4EFgQUFGR5nizi0EuJYW6tIBczyqitSz8wHwYDVR0jBBgwFoAUsEH0
mXTN1Y6EMDthXDINpPxV7XswJwYDVR0RBCAwHoIJbG9jYWxob3N0ghF3d3cubG9j
YWxob3N0Lm5ldDAKBggqhkjOPQQDAgNHADBEAiAIJQfNu4r/6fzp2n2hW7IKe2Ll
ivYxOn7UmNEWAM8KjAIgYVvEyonR3BKTV1UvtwQeasMP68+YcE2LPnEggGiBYbI=
-----END CERTIFICATE-----`)

var firstServerKey = []byte(`-----BEGIN EC PRIVATE KEY-----
MHcCAQEEILShzyTU+kkhmpJQsYvLkd0RpvHz9TNkaVfabJOvorAhoAoGCCqGSM49
AwEHoUQDQgAEXtlWtbQGGXqciklSB4H70SYnt8Vc7U2GweRjKklVT3nKJuVsi91w
VJ7wIgzABwdr/VWXTH3CZOxitQK7Mi09MA==
-----END EC PRIVATE KEY-----`)

var secondCA = []byte(`-----BEGIN CERTIFICATE-----
MIIBrTCCAVSgAwIBAgIUG5LffV/1EZSbB6dZF1mXGHXy3BswCgYIKoZIzj0EAwIw
NTELMAkGA1UEBhMCUEwxDzANBgNVBAcTBkdkYW5zazEVMBMGA1UEAxMMZXhhbXBs
ZTIubmV0MB4XDTIzMTEwNzA4MzYwMFoXDTI4MTEwNTA4MzYwMFowNTELMAkGA1UE
BhMCUEwxDzANBgNVBAcTBkdkYW5zazEVMBMGA1UEAxMMZXhhbXBsZTIubmV0MFkw
EwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE2HijfshtYE1nfKhjxYdw32Ic5fgHU3fg
wRGCA2Qs8XolgFqzrl0AZFvOtvJnO6u5PSbEb49SqtIrINSjolgbp6NCMEAwDgYD
VR0PAQH/BAQDAgEGMA8GA1UdEwEB/wQFMAMBAf8wHQYDVR0OBBYEFHTCLc5SRdG5
bjyBzZ92WgAim5qvMAoGCCqGSM49BAMCA0cAMEQCIG84IsPbB1507Q4QOjZfcnQp
76vglEyUZvJ/cVIgmNaqAiA0k8+WMvmolJ4YCMbOw44QHvMQGXS47Od9UPyfHT5s
fg==
-----END CERTIFICATE-----`)

var secondServerCert = []byte(`-----BEGIN CERTIFICATE-----
MIICDDCCAbOgAwIBAgIUZxpMjqixAG5apDnzZO0SiJK2vEowCgYIKoZIzj0EAwIw
NTELMAkGA1UEBhMCUEwxDzANBgNVBAcTBkdkYW5zazEVMBMGA1UEAxMMZXhhbXBs
ZTIubmV0MB4XDTIzMTEwNzA4MzcwMFoXDTMzMTEwNDA4MzcwMFowNjELMAkGA1UE
BhMCUEwxDzANBgNVBAcTBkdkYW5zazEWMBQGA1UEAxMNbG9jYWxob3N0Lm5ldDBZ
MBMGByqGSM49AgEGCCqGSM49AwEHA0IABA974GeP/JyUL52rg60pF/JtR/B17D5R
G/ciLIB56L8ennkaITt6ePeiflKmVgUfXJWOZHq8opEMMB+H5gdMtLijgZ8wgZww
DgYDVR0PAQH/BAQDAgWgMBMGA1UdJQQMMAoGCCsGAQUFBwMBMAwGA1UdEwEB/wQC
MAAwHQYDVR0OBBYEFFYk6QwIxw4EJTfcXRH8HRm0AFbCMB8GA1UdIwQYMBaAFHTC
Lc5SRdG5bjyBzZ92WgAim5qvMCcGA1UdEQQgMB6CCWxvY2FsaG9zdIIRd3d3Lmxv
Y2FsaG9zdC5uZXQwCgYIKoZIzj0EAwIDRwAwRAIgT3SQeYf/jUlV0YQzVpPab3EQ
lZbrBA9qCcN4SAO/v8QCIBSo8LTI0W4aC1QD+gvzjtGVdKd4JKI6YhoVQk6IpIrP
-----END CERTIFICATE-----
`)

var secondServerKey = []byte(`
-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIEioJF/VzyLYZ0nTMzffnWy0WQwd8d+hy5s7iJChAcOcoAoGCCqGSM49
AwEHoUQDQgAED3vgZ4/8nJQvnauDrSkX8m1H8HXsPlEb9yIsgHnovx6eeRohO3p4
96J+UqZWBR9clY5keryikQwwH4fmB0y0uA==
-----END EC PRIVATE KEY-----
`)

func TestRootCAReloadNotThreadSafe(t *testing.T) {
	stopCh := make(chan struct{})
	// set up the server
	var reloadServerCertificate bool
	firstServerServingCert, err := tls.X509KeyPair(firstServerCert, firstServerKey)
	if err != nil {
		t.Fatalf("server: invalid x509/key pair: %v", err)
	}
	secondServerServingCert, err := tls.X509KeyPair(secondServerCert, secondServerKey)
	if err != nil {
		t.Fatalf("server: invalid x509/key pair: %v", err)
	}
	server := httptest.NewUnstartedServer(nil)
	server.TLS = &tls.Config{
		GetCertificate: func(_ *tls.ClientHelloInfo) (*tls.Certificate, error) {
			if reloadServerCertificate {
				return &secondServerServingCert, nil
			}
			return &firstServerServingCert, nil
		},
		NextProtos: []string{http2.NextProtoTLS},
	}
	server.StartTLS()
	defer server.Close()

	// set up the client for the firstCA
	clientCACertPool := x509.NewCertPool()
	clientCACertPool.AppendCertsFromPEM(firstCA)
	clientTLSConfig := &tls.Config{
		RootCAs: clientCACertPool,
		VerifyPeerCertificate: func(rawCerts [][]byte, verifiedChains [][]*x509.Certificate) error {
			return nil
		},
		ServerName: "localhost",
	}

	client := &http.Client{}
	client.Transport = &http2.Transport{
		TLSClientConfig: clientTLSConfig,
	}

	resp, err := client.Get(fmt.Sprintf("https://127.0.0.1:%d", server.Listener.Addr().(*net.TCPAddr).Port))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	_, err = io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}

	// sleep for a while so that
	// the connection enters the idle phase
	time.Sleep(1 * time.Second)

	// reload the server certificate
	// and tear down the client connection
	reloadServerCertificate = true
	client.CloseIdleConnections()

	resp, err = client.Get(fmt.Sprintf("https://127.0.0.1:%d", server.Listener.Addr().(*net.TCPAddr).Port))
	if err == nil {
		t.Fatal("expected an error, got none")
	}
	// append the new cert to the RootCAs
	//
	// NOTE:
	// this operation is not thread safe
	go func() {
		select {
		case <-stopCh:
			return
		default:
			clientCACertPool.AppendCertsFromPEM(secondCA)
		}
	}()
	defer close(stopCh)

	resp, err = client.Get(fmt.Sprintf("https://127.0.0.1:%d", server.Listener.Addr().(*net.TCPAddr).Port))
	if err != nil {
		t.Fatal(err)
	}
}

func TestVerifyPeerCertificateCallback(t *testing.T) {
	// set up the server
	server := httptest.NewUnstartedServer(nil)
	serverServingCert, err := tls.X509KeyPair(firstServerCert, firstServerKey)
	if err != nil {
		t.Fatalf("server: invalid x509/key pair: %v", err)
	}
	server.TLS = &tls.Config{
		GetCertificate: func(_ *tls.ClientHelloInfo) (*tls.Certificate, error) {
			return &serverServingCert, nil
		},
		NextProtos: []string{http2.NextProtoTLS},
	}
	server.StartTLS()
	defer server.Close()

	// set up the client
	//
	// NOTE:
	// The VerifyPeerCertificate is called only when
	// the InsecureSkipVerify is set to true
	//
	// Setting InsecureSkipVerify to false
	// gives:  tls: failed to verify certificate: x509: certificate signed by unknown authority error
	clientTLSConfig := &tls.Config{
		InsecureSkipVerify: true,
		VerifyPeerCertificate: func(rawCerts [][]byte, verifiedChains [][]*x509.Certificate) error {
			// TODO: verifying the server cert
			return nil
		},
	}

	client := &http.Client{}
	client.Transport = &http2.Transport{
		TLSClientConfig: clientTLSConfig,
	}

	// act
	resp, err := client.Get(fmt.Sprintf("https://127.0.0.1:%d", server.Listener.Addr().(*net.TCPAddr).Port))
	if err != nil {
		t.Fatal(err)
	}

	// validate
	defer resp.Body.Close()
	_, err = io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
}

func TestFirstCAandFirstServerCertValidity(t *testing.T) {
	verifyCertificateValidity(t, firstCA, firstServerCert, firstServerKey)
}

func TestSecondCAandSecondServerCertValidity(t *testing.T) {
	verifyCertificateValidity(t, secondCA, secondServerCert, secondServerKey)
}

func verifyCertificateValidity(t *testing.T, caCert, serverCert, serverKey []byte) {
	// set up the server
	server := httptest.NewUnstartedServer(nil)
	serverServingCert, err := tls.X509KeyPair(serverCert, serverKey)
	if err != nil {
		t.Fatalf("server: invalid x509/key pair: %v", err)
	}
	server.TLS = &tls.Config{
		GetCertificate: func(_ *tls.ClientHelloInfo) (*tls.Certificate, error) {
			return &serverServingCert, nil
		},
		NextProtos: []string{http2.NextProtoTLS},
	}
	server.StartTLS()
	defer server.Close()

	// set up the client
	clientCACertPool := x509.NewCertPool()
	clientCACertPool.AppendCertsFromPEM(caCert)
	clientTLSConfig := &tls.Config{
		RootCAs: clientCACertPool,
		VerifyPeerCertificate: func(rawCerts [][]byte, verifiedChains [][]*x509.Certificate) error {
			return nil
		},
		ServerName: "localhost",
	}

	client := &http.Client{}
	client.Transport = &http2.Transport{
		TLSClientConfig: clientTLSConfig,
	}

	// act
	resp, err := client.Get(fmt.Sprintf("https://127.0.0.1:%d", server.Listener.Addr().(*net.TCPAddr).Port))
	if err != nil {
		t.Fatal(err)
	}

	// validate
	defer resp.Body.Close()
	_, err = io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
}
