// +build example

package main

import (
	"crypto/tls"
	"crypto/x509"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
	"golang.org/x/net/http2"
)

// Example of creating an HTTP Client configured with a client TLS
// certificates. Can be used with endpoints such as HTTPS_PROXY that require
// client certificates.
//
// Requires a cert and key flags, and optionally takes a CA file.
//
// To run:
//   go run -cert <certfile> -key <keyfile> [-ca <cafile>]
//
// You can generate self signed cert and key pem files
// go run $(go env GOROOT)/src/crypto/tls/generate_cert.go -host <hostname>
func main() {
	var clientCertFile, clientKeyFile, caFile string
	flag.StringVar(&clientCertFile, "cert", "cert.pem", "The `certificate file` to load.")
	flag.StringVar(&clientKeyFile, "key", "key.pem", "The `key file` to load.")
	flag.StringVar(&caFile, "ca", "ca.pem", "The `root CA` to load.")
	flag.Parse()

	if len(clientCertFile) == 0 || len(clientKeyFile) == 0 {
		flag.PrintDefaults()
		log.Fatalf("client certificate and key required")
	}

	tlsCfg, err := tlsConfigWithClientCert(clientCertFile, clientKeyFile, caFile)
	if err != nil {
		log.Fatalf("failed to load client cert, %v", err)
	}

	// Copy of net/http.DefaultTransport with TLS config loaded
	tr := defaultHTTPTransport()
	tr.TLSClientConfig = tlsCfg

	// re-enable HTTP/2 because modifing TLS config will prevent auto support
	// for HTTP/2.
	http2.ConfigureTransport(tr)

	// Configure the SDK's session with the HTTP client with TLS client
	// certificate support enabled. This session will be used to create all
	// SDK's API clients.
	sess, err := session.NewSession(&aws.Config{
		HTTPClient: &http.Client{
			Transport: tr,
		},
	})

	// Create each API client will the session configured with the client TLS
	// certificate.
	svc := s3.New(sess)

	resp, err := svc.ListBuckets(&s3.ListBucketsInput{})
	if err != nil {
		log.Fatalf("failed to list buckets, %v", err)
	}

	fmt.Println("Buckets")
	fmt.Println(resp)
}

func tlsConfigWithClientCert(clientCertFile, clientKeyFile, caFile string) (*tls.Config, error) {
	clientCert, err := tls.LoadX509KeyPair(clientCertFile, clientKeyFile)
	if err != nil {
		return nil, fmt.Errorf("unable to load certificat files, %s, %s, %v",
			clientCertFile, clientKeyFile, err)
	}

	tlsCfg := &tls.Config{
		Certificates: []tls.Certificate{
			clientCert,
		},
	}

	if len(caFile) != 0 {
		cert, err := ioutil.ReadFile(caFile)
		if err != nil {
			return nil, fmt.Errorf("unable to load root CA file, %s, %v",
				caFile, err)
		}
		caCertPool := x509.NewCertPool()
		caCertPool.AppendCertsFromPEM(cert)
		tlsCfg.RootCAs = caCertPool
	}

	tlsCfg.BuildNameToCertificate()

	return tlsCfg, nil
}

func defaultHTTPTransport() *http.Transport {
	return &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
			DualStack: true,
		}).DialContext,
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   10, // Increased idle connections per host
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	}
}
