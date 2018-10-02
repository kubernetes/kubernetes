/*
Copyright 2016 The Kubernetes Authors.

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

package server

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/golang/glog"
	"golang.org/x/net/http2"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/validation"
)

const (
	defaultKeepAlivePeriod = 3 * time.Minute
)

// serveSecurely runs the secure http server. It fails only if certificates cannot
// be loaded or the initial listen call fails. The actual server loop (stoppable by closing
// stopCh) runs in a go routine, i.e. serveSecurely does not block.
func (s *SecureServingInfo) Serve(handler http.Handler, shutdownTimeout time.Duration, stopCh <-chan struct{}) error {
	if s.Listener == nil {
		return fmt.Errorf("listener must not be nil")
	}

	secureServer := &http.Server{
		Addr:           s.Listener.Addr().String(),
		Handler:        handler,
		MaxHeaderBytes: 1 << 20,
		TLSConfig: &tls.Config{
			NameToCertificate: s.SNICerts,
			// Can't use SSLv3 because of POODLE and BEAST
			// Can't use TLSv1.0 because of POODLE and BEAST using CBC cipher
			// Can't use TLSv1.1 because of RC4 cipher usage
			MinVersion: tls.VersionTLS12,
			// enable HTTP2 for go's 1.7 HTTP Server
			NextProtos: []string{"h2", "http/1.1"},
		},
	}

	if s.MinTLSVersion > 0 {
		secureServer.TLSConfig.MinVersion = s.MinTLSVersion
	}
	if len(s.CipherSuites) > 0 {
		secureServer.TLSConfig.CipherSuites = s.CipherSuites
	}

	if s.Cert != nil {
		secureServer.TLSConfig.Certificates = []tls.Certificate{*s.Cert}
	}

	// append all named certs. Otherwise, the go tls stack will think no SNI processing
	// is necessary because there is only one cert anyway.
	// Moreover, if ServerCert.CertFile/ServerCert.KeyFile are not set, the first SNI
	// cert will become the default cert. That's what we expect anyway.
	for _, c := range s.SNICerts {
		secureServer.TLSConfig.Certificates = append(secureServer.TLSConfig.Certificates, *c)
	}

	if s.ClientCA != nil {
		// Populate PeerCertificates in requests, but don't reject connections without certificates
		// This allows certificates to be validated by authenticators, while still allowing other auth types
		secureServer.TLSConfig.ClientAuth = tls.RequestClientCert
		// Specify allowed CAs for client certificates
		secureServer.TLSConfig.ClientCAs = s.ClientCA
	}

	// At least 99% of serialized resources in surveyed clusters were smaller than 256kb.
	// This should be big enough to accommodate most API POST requests in a single frame,
	// and small enough to allow a per connection buffer of this size multiplied by `MaxConcurrentStreams`.
	const resourceBody99Percentile = 256 * 1024

	http2Options := &http2.Server{}

	// shrink the per-stream buffer and max framesize from the 1MB default while still accommodating most API POST requests in a single frame
	http2Options.MaxUploadBufferPerStream = resourceBody99Percentile
	http2Options.MaxReadFrameSize = resourceBody99Percentile

	// use the overridden concurrent streams setting or make the default of 250 explicit so we can size MaxUploadBufferPerConnection appropriately
	if s.HTTP2MaxStreamsPerConnection > 0 {
		http2Options.MaxConcurrentStreams = uint32(s.HTTP2MaxStreamsPerConnection)
	} else {
		http2Options.MaxConcurrentStreams = 250
	}

	// increase the connection buffer size from the 1MB default to handle the specified number of concurrent streams
	http2Options.MaxUploadBufferPerConnection = http2Options.MaxUploadBufferPerStream * int32(http2Options.MaxConcurrentStreams)

	// apply settings to the server
	if err := http2.ConfigureServer(secureServer, http2Options); err != nil {
		return fmt.Errorf("error configuring http2: %v", err)
	}

	glog.Infof("Serving securely on %s", secureServer.Addr)
	return RunServer(secureServer, s.Listener, shutdownTimeout, stopCh)
}

// RunServer listens on the given port if listener is not given,
// then spawns a go-routine continuously serving
// until the stopCh is closed. This function does not block.
// TODO: make private when insecure serving is gone from the kube-apiserver
func RunServer(
	server *http.Server,
	ln net.Listener,
	shutDownTimeout time.Duration,
	stopCh <-chan struct{},
) error {
	if ln == nil {
		return fmt.Errorf("listener must not be nil")
	}

	// Shutdown server gracefully.
	go func() {
		<-stopCh
		ctx, cancel := context.WithTimeout(context.Background(), shutDownTimeout)
		server.Shutdown(ctx)
		cancel()
	}()

	go func() {
		defer utilruntime.HandleCrash()

		var listener net.Listener
		listener = tcpKeepAliveListener{ln.(*net.TCPListener)}
		if server.TLSConfig != nil {
			listener = tls.NewListener(listener, server.TLSConfig)
		}

		err := server.Serve(listener)

		msg := fmt.Sprintf("Stopped listening on %s", ln.Addr().String())
		select {
		case <-stopCh:
			glog.Info(msg)
		default:
			panic(fmt.Sprintf("%s due to error: %v", msg, err))
		}
	}()

	return nil
}

type NamedTLSCert struct {
	TLSCert tls.Certificate

	// names is a list of domain patterns: fully qualified domain names, possibly prefixed with
	// wildcard segments.
	Names []string
}

// getNamedCertificateMap returns a map of *tls.Certificate by name. It's is
// suitable for use in tls.Config#NamedCertificates. Returns an error if any of the certs
// cannot be loaded. Returns nil if len(certs) == 0
func GetNamedCertificateMap(certs []NamedTLSCert) (map[string]*tls.Certificate, error) {
	// register certs with implicit names first, reverse order such that earlier trump over the later
	byName := map[string]*tls.Certificate{}
	for i := len(certs) - 1; i >= 0; i-- {
		if len(certs[i].Names) > 0 {
			continue
		}
		cert := &certs[i].TLSCert

		// read names from certificate common names and DNS names
		if len(cert.Certificate) == 0 {
			return nil, fmt.Errorf("empty SNI certificate, skipping")
		}
		x509Cert, err := x509.ParseCertificate(cert.Certificate[0])
		if err != nil {
			return nil, fmt.Errorf("parse error for SNI certificate: %v", err)
		}
		cn := x509Cert.Subject.CommonName
		if cn == "*" || len(validation.IsDNS1123Subdomain(strings.TrimPrefix(cn, "*."))) == 0 {
			byName[cn] = cert
		}
		for _, san := range x509Cert.DNSNames {
			byName[san] = cert
		}
		// intentionally all IPs in the cert are ignored as SNI forbids passing IPs
		// to select a cert. Before go 1.6 the tls happily passed IPs as SNI values.
	}

	// register certs with explicit names last, overwriting every of the implicit ones,
	// again in reverse order.
	for i := len(certs) - 1; i >= 0; i-- {
		namedCert := &certs[i]
		for _, name := range namedCert.Names {
			byName[name] = &certs[i].TLSCert
		}
	}

	return byName, nil
}

// tcpKeepAliveListener sets TCP keep-alive timeouts on accepted
// connections. It's used by ListenAndServe and ListenAndServeTLS so
// dead TCP connections (e.g. closing laptop mid-download) eventually
// go away.
//
// Copied from Go 1.7.2 net/http/server.go
type tcpKeepAliveListener struct {
	*net.TCPListener
}

func (ln tcpKeepAliveListener) Accept() (net.Conn, error) {
	tc, err := ln.AcceptTCP()
	if err != nil {
		return nil, err
	}
	tc.SetKeepAlive(true)
	tc.SetKeepAlivePeriod(defaultKeepAlivePeriod)
	return tc, nil
}
