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
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
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
	s.effectiveSecurePort, err = runServer(secureServer, &s.SecureServingInfo.ServingInfo, stopCh)
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
	s.effectiveInsecurePort, err = runServer(insecureServer, s.InsecureServingInfo, stopCh)
	return err
}

// runServer listens on the given port, then spawns a go-routine continuously serving
// until the stopCh is closed. The port is returned. This function does not block.
func runServer(server *http.Server, info *ServingInfo, stopCh <-chan struct{}) (int, error) {
	if len(server.Addr) == 0 {
		return 0, errors.New("address cannot be empty")
	}

	if len(info.BindNetwork) == 0 {
		info.BindNetwork = "tcp"
	}
	server.ConnState = info.IPLimit.HandleHttpServerStateChange

	// first listen is synchronous (fail early!)
	ln, err := net.Listen(info.BindNetwork, server.Addr)
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
				ipBasedLimitHandler{&info.IPLimit},
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

					ln, err = net.Listen(info.BindNetwork, server.Addr)
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

func (s *ServingInfo) Profile(w http.ResponseWriter, r *http.Request) {
	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "\t")
	err := encoder.Encode(&s.IPLimit)
	if err != nil {
		glog.V(2).Infof("Error marshaling secure server info: %v", err)
	}
}

func NoSecureHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, `{ "Error": "No secure server configured" }`)
}

// tcpConnectionChainListener is responsible to invoking any connection handling
// code which is invoked after AcceptTCP() has been called but before the
// connection is usable by the server for communicating with the client. We
// assume if an error is returned from a chain handler that it will return a nil
// conn and have already closed that conn. Init() is used to allow low levels tweaks
// on the TCPConn.
type tcpConnectionChainHandler interface {
	Init(*net.TCPConn) error
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
		err = handler.Init(tc)
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
type tcpKeepAliveHandler struct{}

func (ln tcpKeepAliveHandler) Init(conn *net.TCPConn) error {
	conn.SetKeepAlive(true)
	conn.SetKeepAlivePeriod(defaultKeepAlivePeriod)
	return nil
}

// IPBasedLimitListener accepts at most n simultaneous connections from a
// given IP address. We also limit the total simultaneous connections.
// None of this is applied to localhost/loopback connections.
//
// Based on Go 1.7.4 net/netutil/listen.go
// TODO: (cheftako) : IPv6, CIDR Range rather than individual IP Addresses?
// Considered using atomic for per IP limit value,
// but problem initing map value safely
type IPBasedLimit struct {
	// The maximum concurent connections an IP is allowed to have open
	MaxPerIP int

	// The total maximum concurrent connections allowed open (Except localhost)
	Max int

	// Lock used to ensure the Limits and total are updated atomicaly
	mutex sync.Mutex

	// Map in which we map the serialized ip to the current count
	Limits map[string]int

	// Map in which we retain the lookup from connection to decrementIPLocked call
	Decrementers map[string]func()

	// Current total count of the connections in existance except those from localhost
	total int
}

func (i *IPBasedLimit) incrementIP(conn, sourceIP string) error {
	if i.Max <= 0 {
		return nil
	}
	ip := net.ParseIP(sourceIP)
	if ip == nil {
		glog.Infof("Received request from client with invalid ip %s", sourceIP)
		return IPLimitExceededError{ip: sourceIP, reason: "unparseable ip address"}
	}
	if ip.IsLoopback() {
		return nil
	}
	i.mutex.Lock()
	defer i.mutex.Unlock()
	current := i.Limits[sourceIP]
	if current >= i.MaxPerIP {
		return IPLimitExceededError{ip: sourceIP, reason: fmt.Sprintf("exceeded per IP limit of %d", i.MaxPerIP)}
	}
	if i.total >= i.Max {
		return IPLimitExceededError{ip: sourceIP, reason: fmt.Sprintf("exceeded overall limit of %d", i.Max)}
	}
	i.total++
	i.Limits[sourceIP] = current + 1
	var releaseOnce sync.Once
	i.Decrementers[conn] = func() {
		releaseOnce.Do(func() { i.decrementIPLocked(conn, sourceIP) })
	}
	return nil
}

// Please note that to call decrementIPLocked you should hold the i.mutex lock.
// Currently the only caller of this method is HandleHttpServerStateChange
func (i *IPBasedLimit) decrementIPLocked(conn, sourceIP string) {
	if i.Max <= 0 {
		return
	}
	ip := net.ParseIP(sourceIP)
	if ip == nil || ip.IsLoopback() {
		return
	}
	i.total--
	current, ok := i.Limits[sourceIP]
	if !ok {
		glog.Errorf("Attempt to decrement limit for not connected IP %s", sourceIP)
	}
	if current > 1 {
		i.Limits[sourceIP] = current - 1
	} else {
		delete(i.Limits, sourceIP)
	}
	delete(i.Decrementers, conn)
}

func (i *IPBasedLimit) HandleHttpServerStateChange(conn net.Conn, state http.ConnState) {
	if state == http.StateClosed || state == http.StateHijacked {
		i.mutex.Lock()
		defer i.mutex.Unlock()
		if dec, ok := i.Decrementers[conn.RemoteAddr().String()]; ok {
			dec()
		}
	}
}

// Using a custom marshal JSON method for several reasons
// A) It allows us to minimize the scope of the mutex which needs to be held
// B) It allows the fields to remain private
// C) It allows us to marshal 'Decrementers' as a list of its keys
// This is only meant for use in the debug API.
func (i *IPBasedLimit) MarshalJSON() ([]byte, error) {
	buffer := bytes.NewBufferString("{")
	fmt.Fprintf(buffer, `"MaxPerIP": %d, `, i.MaxPerIP)
	fmt.Fprintf(buffer, `"MaxTotal": %d, `, i.Max)
	fmt.Fprintf(buffer, `"NumberIPs": %d, `, len(i.Limits))
	fmt.Fprintf(buffer, `"NumberConns": %d, `, len(i.Decrementers))
	fmt.Fprintf(buffer, `"IPCounts": {`)
	first := true
	i.mutex.Lock()
	defer i.mutex.Unlock()
	for ip, count := range i.Limits {
		if first {
			fmt.Fprintf(buffer, `"%s": %d`, ip, count)
			first = false
		} else {
			fmt.Fprintf(buffer, `, "%s": %d`, ip, count)
		}
	}
	buffer.WriteString("}, ")
	buffer.WriteString(` "Connections": [`)
	first = true
	for conn := range i.Decrementers {
		if first {
			fmt.Fprintf(buffer, `"%s"`, conn)
			first = false
		} else {
			fmt.Fprintf(buffer, `, "%s"`, conn)
		}
	}
	buffer.WriteString("] ")
	buffer.WriteString("}")
	return buffer.Bytes(), nil
}

// The actual structure/handler which attaches the limiting code to a connection
type ipBasedLimitHandler struct {
	// The limit structure for this connection type.
	ipLimit *IPBasedLimit
}

func (h ipBasedLimitHandler) Init(conn *net.TCPConn) error {
	remote := conn.RemoteAddr().String()
	ip, _, err := net.SplitHostPort(remote)
	if err != nil {
		conn.Close() // Not decrementing as we have not yet incremented
		glog.Errorf("Error getting remote address for connection: %v", err)
		return err
	}
	err = h.ipLimit.incrementIP(remote, ip)
	if err != nil {
		conn.Close() // Not decrementing as we did not successfully increment
		glog.V(1).Infof("%s. Read the help for --max-requests-inflight if you need to change the limit", err.Error())
		return err
	}
	return nil
}

// A custom error which tells http/server.go Serve() to retry Accept() quickly
type IPLimitExceededError struct {
	// IP which saw the problem, present to allow for reporting and debugging.
	ip     string
	reason string
}

func (l IPLimitExceededError) Error() string {
	return "Accept() blocked connection request from " + l.ip + " because " + l.reason
}
func (l IPLimitExceededError) Timeout() bool   { return true }
func (l IPLimitExceededError) Temporary() bool { return true }
