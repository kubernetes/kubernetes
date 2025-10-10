/*
 *
 * Copyright 2023 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package tlscreds implements mTLS Credentials in xDS Bootstrap File.
// See gRFC A65: github.com/grpc/proposal/blob/master/A65-xds-mtls-creds-in-bootstrap.md.
package tlscreds

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"sync"
	"time"

	"github.com/spiffe/go-spiffe/v2/bundle/spiffebundle"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/tls/certprovider"
	"google.golang.org/grpc/credentials/tls/certprovider/pemfile"
	"google.golang.org/grpc/internal/credentials/spiffe"
	"google.golang.org/grpc/internal/envconfig"
)

// bundle is an implementation of credentials.Bundle which implements mTLS
// Credentials in xDS Bootstrap File.
type bundle struct {
	transportCredentials credentials.TransportCredentials
}

// NewBundle returns a credentials.Bundle which implements mTLS Credentials in xDS
// Bootstrap File. It delegates certificate loading to a file_watcher provider
// if either client certificates or server root CA is specified. The second
// return value is a close func that should be called when the caller no longer
// needs this bundle.
// See gRFC A65: github.com/grpc/proposal/blob/master/A65-xds-mtls-creds-in-bootstrap.md
func NewBundle(jd json.RawMessage) (credentials.Bundle, func(), error) {
	cfg := &struct {
		CertificateFile          string `json:"certificate_file"`
		CACertificateFile        string `json:"ca_certificate_file"`
		PrivateKeyFile           string `json:"private_key_file"`
		SPIFFETrustBundleMapFile string `json:"spiffe_trust_bundle_map_file"`
	}{}

	if jd != nil {
		if err := json.Unmarshal(jd, cfg); err != nil {
			return nil, nil, fmt.Errorf("failed to unmarshal config: %v", err)
		}
	} // Else the config field is absent. Treat it as an empty config.

	if !envconfig.XDSSPIFFEEnabled {
		cfg.SPIFFETrustBundleMapFile = ""
	}
	if cfg.CACertificateFile == "" && cfg.CertificateFile == "" && cfg.PrivateKeyFile == "" && cfg.SPIFFETrustBundleMapFile == "" {
		// We cannot use (and do not need) a file_watcher provider in this case,
		// and can simply directly use the TLS transport credentials.
		// Quoting A65:
		//
		// > The only difference between the file-watcher certificate provider
		// > config and this one is that in the file-watcher certificate
		// > provider, at least one of the "certificate_file" or
		// > "ca_certificate_file" fields must be specified, whereas in this
		// > configuration, it is acceptable to specify neither one.
		// Further, with the introduction of SPIFFE Trust Map support, we also
		// check for this value.
		return &bundle{transportCredentials: credentials.NewTLS(&tls.Config{})}, func() {}, nil
	}
	// Otherwise we need to use a file_watcher provider to watch the CA,
	// private and public keys.

	// The pemfile plugin (file_watcher) currently ignores BuildOptions.
	provider, err := certprovider.GetProvider(pemfile.PluginName, jd, certprovider.BuildOptions{})
	if err != nil {
		return nil, nil, err
	}
	return &bundle{
		transportCredentials: &reloadingCreds{provider: provider},
	}, sync.OnceFunc(func() { provider.Close() }), nil
}

func (t *bundle) TransportCredentials() credentials.TransportCredentials {
	return t.transportCredentials
}

func (t *bundle) PerRPCCredentials() credentials.PerRPCCredentials {
	// mTLS provides transport credentials only. There are no per-RPC
	// credentials.
	return nil
}

func (t *bundle) NewWithMode(string) (credentials.Bundle, error) {
	// This bundle has a single mode which only uses TLS transport credentials,
	// so there is no legitimate case where callers would call NewWithMode.
	return nil, fmt.Errorf("xDS TLS credentials only support one mode")
}

// reloadingCreds is a credentials.TransportCredentials for client
// side mTLS that reloads the server root CA certificate and the client
// certificates from the provider on every client handshake. This is necessary
// because the standard TLS credentials do not support reloading CA
// certificates.
type reloadingCreds struct {
	provider certprovider.Provider
}

func (c *reloadingCreds) ClientHandshake(ctx context.Context, authority string, rawConn net.Conn) (net.Conn, credentials.AuthInfo, error) {
	km, err := c.provider.KeyMaterial(ctx)
	if err != nil {
		return nil, nil, err
	}
	var config *tls.Config
	if km.SPIFFEBundleMap != nil {
		config = &tls.Config{
			InsecureSkipVerify:    true,
			VerifyPeerCertificate: buildSPIFFEVerifyFunc(km.SPIFFEBundleMap),
			Certificates:          km.Certs,
		}
	} else {
		config = &tls.Config{
			RootCAs:      km.Roots,
			Certificates: km.Certs,
		}
	}
	return credentials.NewTLS(config).ClientHandshake(ctx, authority, rawConn)
}

func (c *reloadingCreds) Info() credentials.ProtocolInfo {
	return credentials.ProtocolInfo{SecurityProtocol: "tls"}
}

func (c *reloadingCreds) Clone() credentials.TransportCredentials {
	return &reloadingCreds{provider: c.provider}
}

func (c *reloadingCreds) OverrideServerName(string) error {
	return errors.New("overriding server name is not supported by xDS client TLS credentials")
}

func (c *reloadingCreds) ServerHandshake(net.Conn) (net.Conn, credentials.AuthInfo, error) {
	return nil, nil, errors.New("server handshake is not supported by xDS client TLS credentials")
}

func buildSPIFFEVerifyFunc(spiffeBundleMap map[string]*spiffebundle.Bundle) func(rawCerts [][]byte, verifiedChains [][]*x509.Certificate) error {
	return func(rawCerts [][]byte, _ [][]*x509.Certificate) error {
		rawCertList := make([]*x509.Certificate, len(rawCerts))
		for i, asn1Data := range rawCerts {
			cert, err := x509.ParseCertificate(asn1Data)
			if err != nil {
				return fmt.Errorf("spiffe: verify function could not parse input certificate: %v", err)
			}
			rawCertList[i] = cert
		}
		if len(rawCertList) == 0 {
			return fmt.Errorf("spiffe: verify function has no valid input certificates")
		}
		leafCert := rawCertList[0]
		roots, err := spiffe.GetRootsFromSPIFFEBundleMap(spiffeBundleMap, leafCert)
		if err != nil {
			return err
		}

		opts := x509.VerifyOptions{
			Roots:         roots,
			CurrentTime:   time.Now(),
			Intermediates: x509.NewCertPool(),
		}

		for _, cert := range rawCertList[1:] {
			opts.Intermediates.AddCert(cert)
		}
		// The verified chain is (surprisingly) unused.
		if _, err = rawCertList[0].Verify(opts); err != nil {
			return fmt.Errorf("spiffe: x509 certificate Verify failed: %v", err)
		}
		return nil
	}
}
