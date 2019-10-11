// Copyright 2015 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package transport

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"fmt"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/coreos/etcd/pkg/tlsutil"
)

func NewListener(addr, scheme string, tlsinfo *TLSInfo) (l net.Listener, err error) {
	if l, err = newListener(addr, scheme); err != nil {
		return nil, err
	}
	return wrapTLS(addr, scheme, tlsinfo, l)
}

func newListener(addr string, scheme string) (net.Listener, error) {
	if scheme == "unix" || scheme == "unixs" {
		// unix sockets via unix://laddr
		return NewUnixListener(addr)
	}
	return net.Listen("tcp", addr)
}

func wrapTLS(addr, scheme string, tlsinfo *TLSInfo, l net.Listener) (net.Listener, error) {
	if scheme != "https" && scheme != "unixs" {
		return l, nil
	}
	return newTLSListener(l, tlsinfo, checkSAN)
}

type TLSInfo struct {
	CertFile           string
	KeyFile            string
	CAFile             string // TODO: deprecate this in v4
	TrustedCAFile      string
	ClientCertAuth     bool
	CRLFile            string
	InsecureSkipVerify bool

	// ServerName ensures the cert matches the given host in case of discovery / virtual hosting
	ServerName string

	// HandshakeFailure is optionally called when a connection fails to handshake. The
	// connection will be closed immediately afterwards.
	HandshakeFailure func(*tls.Conn, error)

	// CipherSuites is a list of supported cipher suites.
	// If empty, Go auto-populates it by default.
	// Note that cipher suites are prioritized in the given order.
	CipherSuites []uint16

	selfCert bool

	// parseFunc exists to simplify testing. Typically, parseFunc
	// should be left nil. In that case, tls.X509KeyPair will be used.
	parseFunc func([]byte, []byte) (tls.Certificate, error)

	// AllowedCN is a CN which must be provided by a client.
	AllowedCN string
}

func (info TLSInfo) String() string {
	return fmt.Sprintf("cert = %s, key = %s, ca = %s, trusted-ca = %s, client-cert-auth = %v, crl-file = %s", info.CertFile, info.KeyFile, info.CAFile, info.TrustedCAFile, info.ClientCertAuth, info.CRLFile)
}

func (info TLSInfo) Empty() bool {
	return info.CertFile == "" && info.KeyFile == ""
}

func SelfCert(dirpath string, hosts []string) (info TLSInfo, err error) {
	if err = os.MkdirAll(dirpath, 0700); err != nil {
		return
	}

	certPath := filepath.Join(dirpath, "cert.pem")
	keyPath := filepath.Join(dirpath, "key.pem")
	_, errcert := os.Stat(certPath)
	_, errkey := os.Stat(keyPath)
	if errcert == nil && errkey == nil {
		info.CertFile = certPath
		info.KeyFile = keyPath
		info.selfCert = true
		return
	}

	serialNumberLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	serialNumber, err := rand.Int(rand.Reader, serialNumberLimit)
	if err != nil {
		return
	}

	tmpl := x509.Certificate{
		SerialNumber: serialNumber,
		Subject:      pkix.Name{Organization: []string{"etcd"}},
		NotBefore:    time.Now(),
		NotAfter:     time.Now().Add(365 * (24 * time.Hour)),

		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
	}

	for _, host := range hosts {
		h, _, _ := net.SplitHostPort(host)
		if ip := net.ParseIP(h); ip != nil {
			tmpl.IPAddresses = append(tmpl.IPAddresses, ip)
		} else {
			tmpl.DNSNames = append(tmpl.DNSNames, h)
		}
	}

	priv, err := ecdsa.GenerateKey(elliptic.P521(), rand.Reader)
	if err != nil {
		return
	}

	derBytes, err := x509.CreateCertificate(rand.Reader, &tmpl, &tmpl, &priv.PublicKey, priv)
	if err != nil {
		return
	}

	certOut, err := os.Create(certPath)
	if err != nil {
		return
	}
	pem.Encode(certOut, &pem.Block{Type: "CERTIFICATE", Bytes: derBytes})
	certOut.Close()

	b, err := x509.MarshalECPrivateKey(priv)
	if err != nil {
		return
	}
	keyOut, err := os.OpenFile(keyPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0600)
	if err != nil {
		return
	}
	pem.Encode(keyOut, &pem.Block{Type: "EC PRIVATE KEY", Bytes: b})
	keyOut.Close()

	return SelfCert(dirpath, hosts)
}

func (info TLSInfo) baseConfig() (*tls.Config, error) {
	if info.KeyFile == "" || info.CertFile == "" {
		return nil, fmt.Errorf("KeyFile and CertFile must both be present[key: %v, cert: %v]", info.KeyFile, info.CertFile)
	}

	_, err := tlsutil.NewCert(info.CertFile, info.KeyFile, info.parseFunc)
	if err != nil {
		return nil, err
	}

	cfg := &tls.Config{
		MinVersion: tls.VersionTLS12,
		ServerName: info.ServerName,
	}

	if len(info.CipherSuites) > 0 {
		cfg.CipherSuites = info.CipherSuites
	}

	if info.AllowedCN != "" {
		cfg.VerifyPeerCertificate = func(rawCerts [][]byte, verifiedChains [][]*x509.Certificate) error {
			for _, chains := range verifiedChains {
				if len(chains) != 0 {
					if info.AllowedCN == chains[0].Subject.CommonName {
						return nil
					}
				}
			}
			return errors.New("CommonName authentication failed")
		}
	}

	// this only reloads certs when there's a client request
	// TODO: support server-side refresh (e.g. inotify, SIGHUP), caching
	cfg.GetCertificate = func(clientHello *tls.ClientHelloInfo) (*tls.Certificate, error) {
		return tlsutil.NewCert(info.CertFile, info.KeyFile, info.parseFunc)
	}
	cfg.GetClientCertificate = func(unused *tls.CertificateRequestInfo) (*tls.Certificate, error) {
		return tlsutil.NewCert(info.CertFile, info.KeyFile, info.parseFunc)
	}
	return cfg, nil
}

// cafiles returns a list of CA file paths.
func (info TLSInfo) cafiles() []string {
	cs := make([]string, 0)
	if info.CAFile != "" {
		cs = append(cs, info.CAFile)
	}
	if info.TrustedCAFile != "" {
		cs = append(cs, info.TrustedCAFile)
	}
	return cs
}

// ServerConfig generates a tls.Config object for use by an HTTP server.
func (info TLSInfo) ServerConfig() (*tls.Config, error) {
	cfg, err := info.baseConfig()
	if err != nil {
		return nil, err
	}

	cfg.ClientAuth = tls.NoClientCert
	if info.CAFile != "" || info.ClientCertAuth {
		cfg.ClientAuth = tls.RequireAndVerifyClientCert
	}

	CAFiles := info.cafiles()
	if len(CAFiles) > 0 {
		cp, err := tlsutil.NewCertPool(CAFiles)
		if err != nil {
			return nil, err
		}
		cfg.ClientCAs = cp
	}

	// "h2" NextProtos is necessary for enabling HTTP2 for go's HTTP server
	cfg.NextProtos = []string{"h2"}

	return cfg, nil
}

// ClientConfig generates a tls.Config object for use by an HTTP client.
func (info TLSInfo) ClientConfig() (*tls.Config, error) {
	var cfg *tls.Config
	var err error

	if !info.Empty() {
		cfg, err = info.baseConfig()
		if err != nil {
			return nil, err
		}
	} else {
		cfg = &tls.Config{ServerName: info.ServerName}
	}
	cfg.InsecureSkipVerify = info.InsecureSkipVerify

	CAFiles := info.cafiles()
	if len(CAFiles) > 0 {
		cfg.RootCAs, err = tlsutil.NewCertPool(CAFiles)
		if err != nil {
			return nil, err
		}
	}

	if info.selfCert {
		cfg.InsecureSkipVerify = true
	}
	return cfg, nil
}

// IsClosedConnError returns true if the error is from closing listener, cmux.
// copied from golang.org/x/net/http2/http2.go
func IsClosedConnError(err error) bool {
	// 'use of closed network connection' (Go <=1.8)
	// 'use of closed file or network connection' (Go >1.8, internal/poll.ErrClosing)
	// 'mux: listener closed' (cmux.ErrListenerClosed)
	return err != nil && strings.Contains(err.Error(), "closed")
}
