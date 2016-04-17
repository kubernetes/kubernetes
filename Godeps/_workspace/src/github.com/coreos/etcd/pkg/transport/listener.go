// Copyright 2015 CoreOS, Inc.
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
	"fmt"
	"math/big"
	"net"
	"net/http"
	"os"
	"path"
	"strings"
	"time"

	"github.com/coreos/etcd/pkg/tlsutil"
)

func NewListener(addr string, scheme string, tlscfg *tls.Config) (net.Listener, error) {
	nettype := "tcp"
	if scheme == "unix" {
		// unix sockets via unix://laddr
		nettype = scheme
	}

	l, err := net.Listen(nettype, addr)
	if err != nil {
		return nil, err
	}

	if scheme == "https" {
		if tlscfg == nil {
			return nil, fmt.Errorf("cannot listen on TLS for %s: KeyFile and CertFile are not presented", scheme+"://"+addr)
		}

		l = tls.NewListener(l, tlscfg)
	}

	return l, nil
}

func NewTransport(info TLSInfo, dialtimeoutd time.Duration) (*http.Transport, error) {
	cfg, err := info.ClientConfig()
	if err != nil {
		return nil, err
	}

	t := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		Dial: (&net.Dialer{
			Timeout: dialtimeoutd,
			// value taken from http.DefaultTransport
			KeepAlive: 30 * time.Second,
		}).Dial,
		// value taken from http.DefaultTransport
		TLSHandshakeTimeout: 10 * time.Second,
		TLSClientConfig:     cfg,
	}

	return t, nil
}

type TLSInfo struct {
	CertFile       string
	KeyFile        string
	CAFile         string
	TrustedCAFile  string
	ClientCertAuth bool

	selfCert bool

	// parseFunc exists to simplify testing. Typically, parseFunc
	// should be left nil. In that case, tls.X509KeyPair will be used.
	parseFunc func([]byte, []byte) (tls.Certificate, error)
}

func (info TLSInfo) String() string {
	return fmt.Sprintf("cert = %s, key = %s, ca = %s, trusted-ca = %s, client-cert-auth = %v", info.CertFile, info.KeyFile, info.CAFile, info.TrustedCAFile, info.ClientCertAuth)
}

func (info TLSInfo) Empty() bool {
	return info.CertFile == "" && info.KeyFile == ""
}

func SelfCert(dirpath string, hosts []string) (info TLSInfo, err error) {
	if err = os.MkdirAll(dirpath, 0700); err != nil {
		return
	}

	certPath := path.Join(dirpath, "cert.pem")
	keyPath := path.Join(dirpath, "key.pem")
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
		if ip := net.ParseIP(host); ip != nil {
			tmpl.IPAddresses = append(tmpl.IPAddresses, ip)
		} else {
			tmpl.DNSNames = append(tmpl.DNSNames, strings.Split(host, ":")[0])
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

	tlsCert, err := tlsutil.NewCert(info.CertFile, info.KeyFile, info.parseFunc)
	if err != nil {
		return nil, err
	}

	cfg := &tls.Config{
		Certificates: []tls.Certificate{*tlsCert},
		MinVersion:   tls.VersionTLS10,
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
		cfg = &tls.Config{}
	}

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
