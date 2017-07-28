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
	"context"
	"crypto/tls"
	"fmt"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/golang/glog"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/wait"
	restclient "k8s.io/client-go/rest"
)

// UpdateTransport instruments a restconfig with a transport that dynamically uses
// certificates provided by the manager for TLS client auth.
//
// The config must not already provide an explicit transport.
//
// The returned transport periodically checks the manager to determine if the
// certificate has changed. If it has, the transport shuts down all existing client
// connections, forcing the client to re-handshake with the server and use the
// new certificate.
//
// stopCh should be used to indicate when the transport is unused and doesn't need
// to continue checking the manager.
func UpdateTransport(stopCh <-chan struct{}, clientConfig *restclient.Config, clientCertificateManager Manager) error {
	return updateTransport(stopCh, 10*time.Second, clientConfig, clientCertificateManager)
}

// updateTransport is an internal method that exposes how often this method checks that the
// client cert has changed. Intended for testing.
func updateTransport(stopCh <-chan struct{}, period time.Duration, clientConfig *restclient.Config, clientCertificateManager Manager) error {
	if clientConfig.Transport != nil {
		return fmt.Errorf("there is already a transport configured")
	}
	tlsConfig, err := restclient.TLSConfigFor(clientConfig)
	if err != nil {
		return fmt.Errorf("unable to configure TLS for the rest client: %v", err)
	}
	if tlsConfig == nil {
		tlsConfig = &tls.Config{}
	}
	tlsConfig.Certificates = nil
	tlsConfig.GetClientCertificate = func(requestInfo *tls.CertificateRequestInfo) (*tls.Certificate, error) {
		cert := clientCertificateManager.Current()
		if cert == nil {
			return &tls.Certificate{Certificate: nil}, nil
		}
		return cert, nil
	}

	// Custom dialer that will track all connections it creates.
	t := &connTracker{
		dialer: &net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second},
		conns:  make(map[*closableConn]struct{}),
	}

	lastCert := clientCertificateManager.Current()
	go wait.Until(func() {
		curr := clientCertificateManager.Current()
		if curr == nil || lastCert == curr {
			// Cert hasn't been rotated.
			return
		}
		lastCert = curr

		glog.Infof("certificate rotation detected, shutting down client connections to start using new credentials")
		// The cert has been rotated. Close all existing connections to force the client
		// to reperform its TLS handshake with new cert.
		//
		// See: https://github.com/kubernetes-incubator/bootkube/pull/663#issuecomment-318506493
		t.closeAllConns()
	}, period, stopCh)

	clientConfig.Transport = utilnet.SetTransportDefaults(&http.Transport{
		Proxy:               http.ProxyFromEnvironment,
		TLSHandshakeTimeout: 10 * time.Second,
		TLSClientConfig:     tlsConfig,
		MaxIdleConnsPerHost: 25,
		DialContext:         t.DialContext, // Use custom dialer.
	})

	// Zero out all existing TLS options since our new transport enforces them.
	clientConfig.CertData = nil
	clientConfig.KeyData = nil
	clientConfig.CertFile = ""
	clientConfig.KeyFile = ""
	clientConfig.CAData = nil
	clientConfig.CAFile = ""
	clientConfig.Insecure = false
	return nil
}

// connTracker is a dialer that tracks all open connections it creates.
type connTracker struct {
	dialer *net.Dialer

	mu    sync.Mutex
	conns map[*closableConn]struct{}
}

// closeAllConns forcibly closes all tracked connections.
func (c *connTracker) closeAllConns() {
	c.mu.Lock()
	conns := c.conns
	c.conns = make(map[*closableConn]struct{})
	c.mu.Unlock()

	for conn := range conns {
		conn.Close()
	}
}

func (c *connTracker) DialContext(ctx context.Context, network, address string) (net.Conn, error) {
	conn, err := c.dialer.DialContext(ctx, network, address)
	if err != nil {
		return nil, err
	}

	closable := &closableConn{Conn: conn}

	// Start tracking the connection
	c.mu.Lock()
	c.conns[closable] = struct{}{}
	c.mu.Unlock()

	// When the connection is closed, remove it from the map. This will
	// be no-op if the connection isn't in the map, e.g. if closeAllConns()
	// is called.
	closable.onClose = func() {
		c.mu.Lock()
		delete(c.conns, closable)
		c.mu.Unlock()
	}

	return closable, nil
}

type closableConn struct {
	onClose func()
	net.Conn
}

func (c *closableConn) Close() error {
	go c.onClose()
	return c.Conn.Close()
}
