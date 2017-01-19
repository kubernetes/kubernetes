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
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/validation"

	"github.com/golang/glog"
	"github.com/pkg/errors"
)

const (
	defaultKeepAlivePeriod = 3 * time.Minute
)

// serveSecurely runs the secure http server. It fails only if certificates cannot
// be loaded or the initial listen call fails. The actual server loop (stoppable by closing
// stopCh) runs in a go routine, i.e. serveSecurely does not block.
func (s *GenericAPIServer) serveSecurely(stopCh <-chan struct{}) error {
	secureServer := &http.Server{
		Addr:           s.SecureServingInfo.BindAddress,
		Handler:        s.Handler,
		MaxHeaderBytes: 1 << 20,
		TLSConfig: &tls.Config{
			NameToCertificate: s.SecureServingInfo.SNICerts,
			// Can't use SSLv3 because of POODLE and BEAST
			// Can't use TLSv1.0 because of POODLE and BEAST using CBC cipher
			// Can't use TLSv1.1 because of RC4 cipher usage
			MinVersion: tls.VersionTLS12,
			// enable HTTP2 for go's 1.7 HTTP Server
			NextProtos: []string{"h2", "http/1.1"},
		},
	}

	if s.SecureServingInfo.Cert != nil {
		secureServer.TLSConfig.Certificates = []tls.Certificate{*s.SecureServingInfo.Cert}
	}

	// append all named certs. Otherwise, the go tls stack will think no SNI processing
	// is necessary because there is only one cert anyway.
	// Moreover, if ServerCert.CertFile/ServerCert.KeyFile are not set, the first SNI
	// cert will become the default cert. That's what we expect anyway.
	for _, c := range s.SecureServingInfo.SNICerts {
		secureServer.TLSConfig.Certificates = append(secureServer.TLSConfig.Certificates, *c)
	}

	if s.SecureServingInfo.ClientCA != nil {
		// Populate PeerCertificates in requests, but don't reject connections without certificates
		// This allows certificates to be validated by authenticators, while still allowing other auth types
		secureServer.TLSConfig.ClientAuth = tls.RequestClientCert
		// Specify allowed CAs for client certificates
		secureServer.TLSConfig.ClientCAs = s.SecureServingInfo.ClientCA
	}

	glog.Infof("Serving securely on %s", s.SecureServingInfo.BindAddress)
	var err error
	s.effectiveSecurePort, err = runServer(secureServer, s.SecureServingInfo.BindNetwork, &s.SecureServingInfo.ipLimit, stopCh)
	return err
}

// serveInsecurely run the insecure http server. It fails only if the initial listen
// call fails. The actual server loop (stoppable by closing stopCh) runs in a go
// routine, i.e. serveInsecurely does not block.
func (s *GenericAPIServer) serveInsecurely(stopCh <-chan struct{}) error {
	insecureServer := &http.Server{
		Addr:           s.InsecureServingInfo.BindAddress,
		Handler:        s.InsecureHandler,
		MaxHeaderBytes: 1 << 20,
	}
	glog.Infof("Serving insecurely on %s", s.InsecureServingInfo.BindAddress)
	var err error
	s.effectiveInsecurePort, err = runServer(insecureServer, s.InsecureServingInfo.BindNetwork, &s.InsecureServingInfo.ipLimit, stopCh)
	return err
}

// runServer listens on the given port, then spawns a go-routine continuously serving
// until the stopCh is closed. The port is returned. This function does not block.
func runServer(server *http.Server, network string, ipLimit *ipBasedLimit, stopCh <-chan struct{}) (int, error) {
	if len(server.Addr) == 0 {
		return 0, errors.New("address cannot be empty")
	}

	if len(network) == 0 {
		network = "tcp"
	}

	// first listen is synchronous (fail early!)
	ln, err := net.Listen(network, server.Addr)
	if err != nil {
		return 0, fmt.Errorf("failed to listen on %v: %v", server.Addr, err)
	}

	// get port
	tcpAddr, ok := ln.Addr().(*net.TCPAddr)
	if !ok {
		ln.Close()
		return 0, fmt.Errorf("invalid listen address: %q", ln.Addr().String())
	}

	lock := sync.Mutex{} // to avoid we close an old listener during a listen retry
	go func() {
		<-stopCh
		lock.Lock()
		defer lock.Unlock()
		ln.Close()
	}()

	go func() {
		defer utilruntime.HandleCrash()

		for {
			var listener net.Listener
			chain := []tcpConnectionChainHandler{
				tcpKeepAliveHandler{},
				ipBasedLimitHandler{ipLimit},
			}
			listener = tcpConnectionChainListener{ln.(*net.TCPListener), chain}
			if server.TLSConfig != nil {
				listener = tls.NewListener(listener, server.TLSConfig)
			}

			err := server.Serve(listener)
			glog.Errorf("Error serving %v (%v); will try again.", server.Addr, err)

			// listen again
			func() {
				lock.Lock()
				defer lock.Unlock()
				for {
					time.Sleep(15 * time.Second)

					ln, err = net.Listen("tcp", server.Addr)
					if err == nil {
						return
					}
					select {
					case <-stopCh:
						return
					default:
					}
					glog.Errorf("Error listening on %v (%v); will try again.", server.Addr, err)
				}
			}()

			select {
			case <-stopCh:
				return
			default:
			}
		}
	}()

	return tcpAddr.Port, nil
}

type namedTlsCert struct {
	tlsCert tls.Certificate

	// names is a list of domain patterns: fully qualified domain names, possibly prefixed with
	// wildcard segments.
	names []string
}

// getNamedCertificateMap returns a map of *tls.Certificate by name. It's is
// suitable for use in tls.Config#NamedCertificates. Returns an error if any of the certs
// cannot be loaded. Returns nil if len(certs) == 0
func getNamedCertificateMap(certs []namedTlsCert) (map[string]*tls.Certificate, error) {
	// register certs with implicit names first, reverse order such that earlier trump over the later
	byName := map[string]*tls.Certificate{}
	for i := len(certs) - 1; i >= 0; i-- {
		if len(certs[i].names) > 0 {
			continue
		}
		cert := &certs[i].tlsCert

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
		for _, name := range namedCert.names {
			byName[name] = &certs[i].tlsCert
		}
	}

	return byName, nil
}

// tcpConnectionChainListener is responsible to invoking any connection handling
// code which is invoked after AcceptTCP() has been called but before the
// connection is usable by the server for communicating with the client. We
// assume if an error is returned from a chain handler that it will return a nil
// conn and have already closed that conn.
type tcpConnectionChainHandler interface {
	init(*net.TCPConn) error
	wrap(net.Conn) (net.Conn, error)
}

type tcpConnectionChainListener struct {
	*net.TCPListener
	chain []tcpConnectionChainHandler
}

func (l tcpConnectionChainListener) Accept() (net.Conn, error) {
	tc, err := l.AcceptTCP()
	if err != nil {
		return nil, err
	}
	for _, handler := range l.chain {
		err = handler.init(tc)
		if err != nil {
			return nil, err
		}
	}
	var conn net.Conn = tc
	for _, handler := range l.chain {
		conn, err = handler.wrap(conn)
		if err != nil {
			return nil, err
		}
	}
	return tc, nil
}

// tcpKeepAliveListener sets TCP keep-alive timeouts on accepted
// connections. It's used by ListenAndServe and ListenAndServeTLS so
// dead TCP connections (e.g. closing laptop mid-download) eventually
// go away.
//
// Copied from Go 1.7.2 net/http/server.go
type tcpKeepAliveHandler struct {
}

func (ln tcpKeepAliveHandler) init(conn *net.TCPConn) error {
	conn.SetKeepAlive(true)
	conn.SetKeepAlivePeriod(defaultKeepAlivePeriod)
	return nil
}

func (ln tcpKeepAliveHandler) wrap(conn net.Conn) (net.Conn, error) {
	return conn, nil
}

// ipBasedLimitListener accepts at most n simultaneous connections from a
// given IP address. We also limit the total simultaneous connections.
// None of this is applied to localhost/loopback connections.
//
// Based on Go 1.7.4 net/netutil/listen.go
// TODO: (cheftako) : IPv6, CIDR Range rather than individual IP Addresses?
// Considered using atomic for per IP limit value,
// but problem initing map value safely
type ipBasedLimit struct {
	// Map in which we map the serialized ip to the current count
	limits map[string]int

	// Current total count of the connections in existance (no localhost)
	total int

	// Lock used to ensure the limits and total are updated atomicaly
	mutex sync.Mutex

	// The maximum concurent connections an IP is allowed to have open
	ipMax int

	// The total maximum concurrent connections allowed open (Except localhost)
	max int
}

func (i *ipBasedLimit) incrementIp(ipser string) error {
	if i.max <= 0 {
		return nil
	}
	ip := net.ParseIP(ipser)
	if ip == nil {
		glog.Infof("Received request from client with invalid ip %s", ipser)
		return limitedError{ip: ipser}
	}
	if ip.IsLoopback() {
		return nil
	}
	i.mutex.Lock()
	defer i.mutex.Unlock()
	current := i.limits[ipser]
	if current >= i.ipMax || i.total >= i.max {
		return limitedError{ip: ipser}
	}
	i.total++
	i.limits[ipser] = current + 1
	return nil
}

func (i *ipBasedLimit) decrementIp(ipser string) {
	if i.max <= 0 {
		return
	}
	ip := net.ParseIP(ipser)
	if ip == nil || ip.IsLoopback() {
		return
	}
	i.mutex.Lock()
	defer i.mutex.Unlock()
	i.total--
	current := i.limits[ipser]
	if current > 1 {
		i.limits[ipser] = current - 1
	} else {
		glog.Infof("Cleaned up all references for IP %s", ipser)
		delete(i.limits, ipser)
	}
}

// The actual structure/handler which attaches the limiting code to a connection
type ipBasedLimitHandler struct {
	// The limit structure for this connection type.
	ipLimit *ipBasedLimit
}

func (h ipBasedLimitHandler) init(conn *net.TCPConn) error {
	return nil
}

func (h ipBasedLimitHandler) wrap(conn net.Conn) (net.Conn, error) {
	ip, _, err := net.SplitHostPort(conn.RemoteAddr().String())
	if err != nil {
		conn.Close() // Not decrementing as we have not yet incremented
		glog.Infof("Something went wrong getting the remote address %v", err)
		return nil, err
	}
	local := conn.LocalAddr().String()
	glog.Infof("Entered ip based limit Accept() for IP %s on %s", ip, local)
	err = h.ipLimit.incrementIp(ip)
	if err != nil {
		conn.Close() // Not decrementing as we did not successfully increment
		glog.Infof("Denied accept to IP %s", ip)
		return nil, err
	}
	glog.Infof("Allowed accept to IP %s", ip)
	return &ipBasedLimitListenerConn{Conn: conn, ipLimit: h.ipLimit, ip: ip}, nil
}

// The structure which is responsible for cleaning up/decrementing the related ip limit
type ipBasedLimitListenerConn struct {
	// The underlying connection used to for the request/response.
	net.Conn

	// Makes sure we do not decrement more than once for the given connection
	releaseOnce sync.Once

	// Reference to the relevant IP Limiting structure.
	ipLimit *ipBasedLimit

	// The IP addess used when reserving/incrementIp so we can clean up
	ip string
}

func (c *ipBasedLimitListenerConn) Close() error {
	err := c.Conn.Close()
	c.releaseOnce.Do(func() { c.ipLimit.decrementIp(c.ip) })
	return err
}

// A custom error which tells http/server.go Serve() to retry Accept() quickly
type limitedError struct {
	// Present for debugging - IP which saw the problem
	ip string
}

func (l limitedError) Error() string {
	return "Accept() Exceeded maximum concurrent requests for IP " + l.ip
}
func (l limitedError) Timeout() bool   { return true }
func (l limitedError) Temporary() bool { return true }
