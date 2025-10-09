/*
 *
 * Copyright 2020 gRPC authors.
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

// Package xds contains non-user facing functionality of the xds credentials.
package xds

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"strings"
	"unsafe"

	"google.golang.org/grpc/attributes"
	"google.golang.org/grpc/credentials/tls/certprovider"
	"google.golang.org/grpc/internal"
	"google.golang.org/grpc/internal/credentials/spiffe"
	"google.golang.org/grpc/internal/xds/matcher"
	"google.golang.org/grpc/resolver"
)

func init() {
	internal.GetXDSHandshakeInfoForTesting = GetHandshakeInfo
}

// handshakeAttrKey is the type used as the key to store HandshakeInfo in
// the Attributes field of resolver.Address.
type handshakeAttrKey struct{}

// Equal reports whether the handshake info structs are identical.
func (hi *HandshakeInfo) Equal(other *HandshakeInfo) bool {
	if hi == nil && other == nil {
		return true
	}
	if hi == nil || other == nil {
		return false
	}
	if hi.rootProvider != other.rootProvider ||
		hi.identityProvider != other.identityProvider ||
		hi.requireClientCert != other.requireClientCert ||
		len(hi.sanMatchers) != len(other.sanMatchers) {
		return false
	}
	for i := range hi.sanMatchers {
		if !hi.sanMatchers[i].Equal(other.sanMatchers[i]) {
			return false
		}
	}
	return true
}

// SetHandshakeInfo returns a copy of addr in which the Attributes field is
// updated with hiPtr.
func SetHandshakeInfo(addr resolver.Address, hiPtr *unsafe.Pointer) resolver.Address {
	addr.Attributes = addr.Attributes.WithValue(handshakeAttrKey{}, hiPtr)
	return addr
}

// GetHandshakeInfo returns a pointer to the *HandshakeInfo stored in attr.
func GetHandshakeInfo(attr *attributes.Attributes) *unsafe.Pointer {
	v := attr.Value(handshakeAttrKey{})
	hi, _ := v.(*unsafe.Pointer)
	return hi
}

// HandshakeInfo wraps all the security configuration required by client and
// server handshake methods in xds credentials. The xDS implementation will be
// responsible for populating these fields.
type HandshakeInfo struct {
	// All fields written at init time and read only after that, so no
	// synchronization needed.
	rootProvider      certprovider.Provider
	identityProvider  certprovider.Provider
	sanMatchers       []matcher.StringMatcher // Only on the client side.
	requireClientCert bool                    // Only on server side.
}

// NewHandshakeInfo returns a new handshake info configured with the provided
// options.
func NewHandshakeInfo(rootProvider certprovider.Provider, identityProvider certprovider.Provider, sanMatchers []matcher.StringMatcher, requireClientCert bool) *HandshakeInfo {
	return &HandshakeInfo{
		rootProvider:      rootProvider,
		identityProvider:  identityProvider,
		sanMatchers:       sanMatchers,
		requireClientCert: requireClientCert,
	}
}

// UseFallbackCreds returns true when fallback credentials are to be used based
// on the contents of the HandshakeInfo.
func (hi *HandshakeInfo) UseFallbackCreds() bool {
	if hi == nil {
		return true
	}
	return hi.identityProvider == nil && hi.rootProvider == nil
}

// GetSANMatchersForTesting returns the SAN matchers stored in HandshakeInfo.
// To be used only for testing purposes.
func (hi *HandshakeInfo) GetSANMatchersForTesting() []matcher.StringMatcher {
	return append([]matcher.StringMatcher{}, hi.sanMatchers...)
}

// ClientSideTLSConfig constructs a tls.Config to be used in a client-side
// handshake based on the contents of the HandshakeInfo.
func (hi *HandshakeInfo) ClientSideTLSConfig(ctx context.Context) (*tls.Config, error) {
	// On the client side, rootProvider is mandatory. IdentityProvider is
	// optional based on whether the client is doing TLS or mTLS.
	if hi.rootProvider == nil {
		return nil, errors.New("xds: CertificateProvider to fetch trusted roots is missing, cannot perform TLS handshake. Please check configuration on the management server")
	}
	// Since the call to KeyMaterial() can block, we read the providers under
	// the lock but call the actual function after releasing the lock.
	rootProv, idProv := hi.rootProvider, hi.identityProvider

	// InsecureSkipVerify needs to be set to true because we need to perform
	// custom verification to check the SAN on the received certificate.
	// Currently the Go stdlib does complete verification of the cert (which
	// includes hostname verification) or none. We are forced to go with the
	// latter and perform the normal cert validation ourselves.
	cfg := &tls.Config{
		InsecureSkipVerify: true,
		NextProtos:         []string{"h2"},
	}

	km, err := rootProv.KeyMaterial(ctx)
	if err != nil {
		return nil, fmt.Errorf("xds: fetching trusted roots from CertificateProvider failed: %v", err)
	}
	cfg.RootCAs = km.Roots
	cfg.VerifyPeerCertificate = hi.buildVerifyFunc(km, true)

	if idProv != nil {
		km, err := idProv.KeyMaterial(ctx)
		if err != nil {
			return nil, fmt.Errorf("xds: fetching identity certificates from CertificateProvider failed: %v", err)
		}
		cfg.Certificates = km.Certs
	}
	return cfg, nil
}

func (hi *HandshakeInfo) buildVerifyFunc(km *certprovider.KeyMaterial, isClient bool) func(rawCerts [][]byte, _ [][]*x509.Certificate) error {
	return func(rawCerts [][]byte, _ [][]*x509.Certificate) error {
		// Parse all raw certificates presented by the peer.
		var certs []*x509.Certificate
		for _, rc := range rawCerts {
			cert, err := x509.ParseCertificate(rc)
			if err != nil {
				return err
			}
			certs = append(certs, cert)
		}

		// Build the intermediates list and verify that the leaf certificate is
		// signed by one of the root certificates. If a SPIFFE Bundle Map is
		// configured, it is used to get the root certs. Otherwise, the
		// configured roots in the root provider are used.
		intermediates := x509.NewCertPool()
		for _, cert := range certs[1:] {
			intermediates.AddCert(cert)
		}
		roots := km.Roots
		// If a SPIFFE Bundle Map is configured, find the roots for the trust
		// domain of the leaf certificate.
		if km.SPIFFEBundleMap != nil {
			var err error
			roots, err = spiffe.GetRootsFromSPIFFEBundleMap(km.SPIFFEBundleMap, certs[0])
			if err != nil {
				return err
			}
		}
		opts := x509.VerifyOptions{
			Roots:         roots,
			Intermediates: intermediates,
			KeyUsages:     []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		}
		if isClient {
			opts.KeyUsages = []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth}
		} else {
			opts.KeyUsages = []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth}
		}
		if _, err := certs[0].Verify(opts); err != nil {
			return err
		}
		// The SANs sent by the MeshCA are encoded as SPIFFE IDs. We need to
		// only look at the SANs on the leaf cert.
		if cert := certs[0]; !hi.MatchingSANExists(cert) {
			// TODO: Print the complete certificate once the x509 package
			// supports a String() method on the Certificate type.
			return fmt.Errorf("xds: received SANs {DNSNames: %v, EmailAddresses: %v, IPAddresses: %v, URIs: %v} do not match any of the accepted SANs", cert.DNSNames, cert.EmailAddresses, cert.IPAddresses, cert.URIs)
		}
		return nil
	}
}

// ServerSideTLSConfig constructs a tls.Config to be used in a server-side
// handshake based on the contents of the HandshakeInfo.
func (hi *HandshakeInfo) ServerSideTLSConfig(ctx context.Context) (*tls.Config, error) {
	cfg := &tls.Config{
		ClientAuth: tls.NoClientCert,
		NextProtos: []string{"h2"},
	}
	// On the server side, identityProvider is mandatory. RootProvider is
	// optional based on whether the server is doing TLS or mTLS.
	if hi.identityProvider == nil {
		return nil, errors.New("xds: CertificateProvider to fetch identity certificate is missing, cannot perform TLS handshake. Please check configuration on the management server")
	}
	// Since the call to KeyMaterial() can block, we read the providers under
	// the lock but call the actual function after releasing the lock.
	rootProv, idProv := hi.rootProvider, hi.identityProvider
	if hi.requireClientCert {
		cfg.ClientAuth = tls.RequireAndVerifyClientCert
	}

	// identityProvider is mandatory on the server side.
	km, err := idProv.KeyMaterial(ctx)
	if err != nil {
		return nil, fmt.Errorf("xds: fetching identity certificates from CertificateProvider failed: %v", err)
	}
	cfg.Certificates = km.Certs

	if rootProv != nil {
		km, err := rootProv.KeyMaterial(ctx)
		if err != nil {
			return nil, fmt.Errorf("xds: fetching trusted roots from CertificateProvider failed: %v", err)
		}
		if km.SPIFFEBundleMap != nil && hi.requireClientCert {
			// ClientAuth, if set greater than tls.RequireAnyClientCert, must be
			// dropped to tls.RequireAnyClientCert so that custom verification
			// to use SPIFFE Bundles is done.
			cfg.ClientAuth = tls.RequireAnyClientCert
			cfg.VerifyPeerCertificate = hi.buildVerifyFunc(km, false)
		} else {
			cfg.ClientCAs = km.Roots
		}
	}
	return cfg, nil
}

// MatchingSANExists returns true if the SANs contained in cert match the
// criteria enforced by the list of SAN matchers in HandshakeInfo.
//
// If the list of SAN matchers in the HandshakeInfo is empty, this function
// returns true for all input certificates.
func (hi *HandshakeInfo) MatchingSANExists(cert *x509.Certificate) bool {
	if len(hi.sanMatchers) == 0 {
		return true
	}

	// SANs can be specified in any of these four fields on the parsed cert.
	for _, san := range cert.DNSNames {
		if hi.matchSAN(san, true) {
			return true
		}
	}
	for _, san := range cert.EmailAddresses {
		if hi.matchSAN(san, false) {
			return true
		}
	}
	for _, san := range cert.IPAddresses {
		if hi.matchSAN(san.String(), false) {
			return true
		}
	}
	for _, san := range cert.URIs {
		if hi.matchSAN(san.String(), false) {
			return true
		}
	}
	return false
}

// Caller must hold mu.
func (hi *HandshakeInfo) matchSAN(san string, isDNS bool) bool {
	for _, matcher := range hi.sanMatchers {
		if em := matcher.ExactMatch(); em != "" && isDNS {
			// This is a special case which is documented in the xDS protos.
			// If the DNS SAN is a wildcard entry, and the match criteria is
			// `exact`, then we need to perform DNS wildcard matching
			// instead of regular string comparison.
			if dnsMatch(em, san) {
				return true
			}
			continue
		}
		if matcher.Match(san) {
			return true
		}
	}
	return false
}

// dnsMatch implements a DNS wildcard matching algorithm based on RFC2828 and
// grpc-java's implementation in `OkHostnameVerifier` class.
//
// NOTE: Here the `host` argument is the one from the set of string matchers in
// the xDS proto and the `san` argument is a DNS SAN from the certificate, and
// this is the one which can potentially contain a wildcard pattern.
func dnsMatch(host, san string) bool {
	// Add trailing "." and turn them into absolute domain names.
	if !strings.HasSuffix(host, ".") {
		host += "."
	}
	if !strings.HasSuffix(san, ".") {
		san += "."
	}
	// Domain names are case-insensitive.
	host = strings.ToLower(host)
	san = strings.ToLower(san)

	// If san does not contain a wildcard, do exact match.
	if !strings.Contains(san, "*") {
		return host == san
	}

	// Wildcard dns matching rules
	// - '*' is only permitted in the left-most label and must be the only
	//   character in that label. For example, *.example.com is permitted, while
	//   *a.example.com, a*.example.com, a*b.example.com, a.*.example.com are
	//   not permitted.
	// - '*' matches a single domain name component. For example, *.example.com
	//   matches test.example.com but does not match sub.test.example.com.
	// - Wildcard patterns for single-label domain names are not permitted.
	if san == "*." || !strings.HasPrefix(san, "*.") || strings.Contains(san[1:], "*") {
		return false
	}
	// Optimization: at this point, we know that the san contains a '*' and
	// is the first domain component of san. So, the host name must be at
	// least as long as the san to be able to match.
	if len(host) < len(san) {
		return false
	}
	// Hostname must end with the non-wildcard portion of san.
	if !strings.HasSuffix(host, san[1:]) {
		return false
	}
	// At this point we know that the hostName and san share the same suffix
	// (the non-wildcard portion of san). Now, we just need to make sure
	// that the '*' does not match across domain components.
	hostPrefix := strings.TrimSuffix(host, san[1:])
	return !strings.Contains(hostPrefix, ".")
}
