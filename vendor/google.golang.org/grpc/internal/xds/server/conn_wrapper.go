/*
 *
 * Copyright 2021 gRPC authors.
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

package server

import (
	"fmt"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc/credentials/tls/certprovider"
	xdsinternal "google.golang.org/grpc/internal/credentials/xds"
	"google.golang.org/grpc/internal/transport"
	"google.golang.org/grpc/internal/xds/xdsclient/xdsresource"
)

// connWrapper is a thin wrapper around a net.Conn returned by Accept(). It
// provides the following additional functionality:
//  1. A way to retrieve the configured deadline. This is required by the
//     ServerHandshake() method of the xdsCredentials when it attempts to read
//     key material from the certificate providers.
//  2. Implements the XDSHandshakeInfo() method used by the xdsCredentials to
//     retrieve the configured certificate providers.
//  3. xDS filter_chain configuration determines security configuration.
//  4. Dynamically reads routing configuration in UsableRouteConfiguration(), called
//     to process incoming RPC's. (LDS + RDS configuration).
type connWrapper struct {
	net.Conn

	// The specific filter chain picked for handling this connection.
	filterChain *xdsresource.FilterChain

	// A reference to the listenerWrapper on which this connection was accepted.
	parent *listenerWrapper

	// The certificate providers created for this connection.
	rootProvider, identityProvider certprovider.Provider

	// The connection deadline as configured by the grpc.Server on the rawConn
	// that is returned by a call to Accept(). This is set to the connection
	// timeout value configured by the user (or to a default value) before
	// initiating the transport credential handshake, and set to zero after
	// completing the HTTP2 handshake.
	deadlineMu sync.Mutex
	deadline   time.Time

	mu       sync.Mutex
	st       transport.ServerTransport
	draining bool

	// The virtual hosts with matchable routes and instantiated HTTP Filters per
	// route, or an error.
	urc *atomic.Pointer[xdsresource.UsableRouteConfiguration]
}

// UsableRouteConfiguration returns the UsableRouteConfiguration to be used for
// server side routing.
func (c *connWrapper) UsableRouteConfiguration() xdsresource.UsableRouteConfiguration {
	return *c.urc.Load()
}

// SetDeadline makes a copy of the passed in deadline and forwards the call to
// the underlying rawConn.
func (c *connWrapper) SetDeadline(t time.Time) error {
	c.deadlineMu.Lock()
	c.deadline = t
	c.deadlineMu.Unlock()
	return c.Conn.SetDeadline(t)
}

// GetDeadline returns the configured deadline. This will be invoked by the
// ServerHandshake() method of the XdsCredentials, which needs a deadline to
// pass to the certificate provider.
func (c *connWrapper) GetDeadline() time.Time {
	c.deadlineMu.Lock()
	t := c.deadline
	c.deadlineMu.Unlock()
	return t
}

// XDSHandshakeInfo returns a HandshakeInfo with appropriate security
// configuration for this connection. This method is invoked by the
// ServerHandshake() method of the XdsCredentials.
func (c *connWrapper) XDSHandshakeInfo() (*xdsinternal.HandshakeInfo, error) {
	if c.filterChain.SecurityCfg == nil {
		// If the security config is empty, this means that the control plane
		// did not provide any security configuration and therefore we should
		// return an empty HandshakeInfo here so that the xdsCreds can use the
		// configured fallback credentials.
		return xdsinternal.NewHandshakeInfo(nil, nil, nil, false), nil
	}

	cpc := c.parent.xdsC.BootstrapConfig().CertProviderConfigs()
	// Identity provider name is mandatory on the server-side, and this is
	// enforced when the resource is received at the XDSClient layer.
	secCfg := c.filterChain.SecurityCfg
	ip, err := buildProviderFunc(cpc, secCfg.IdentityInstanceName, secCfg.IdentityCertName, true, false)
	if err != nil {
		return nil, err
	}
	// Root provider name is optional and required only when doing mTLS.
	var rp certprovider.Provider
	if instance, cert := secCfg.RootInstanceName, secCfg.RootCertName; instance != "" {
		rp, err = buildProviderFunc(cpc, instance, cert, false, true)
		if err != nil {
			return nil, err
		}
	}
	c.identityProvider = ip
	c.rootProvider = rp

	return xdsinternal.NewHandshakeInfo(c.rootProvider, c.identityProvider, nil, secCfg.RequireClientCert), nil
}

// PassServerTransport drains the passed in ServerTransport if draining is set,
// or persists it to be drained once drained is called.
func (c *connWrapper) PassServerTransport(st transport.ServerTransport) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.draining {
		st.Drain("draining")
	} else {
		c.st = st
	}
}

// Drain drains the associated ServerTransport, or sets draining to true so it
// will be drained after it is created.
func (c *connWrapper) Drain() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.st == nil {
		c.draining = true
	} else {
		c.st.Drain("draining")
	}
}

// Close closes the providers and the underlying connection.
func (c *connWrapper) Close() error {
	if c.identityProvider != nil {
		c.identityProvider.Close()
	}
	if c.rootProvider != nil {
		c.rootProvider.Close()
	}
	c.parent.removeConn(c)
	return c.Conn.Close()
}

func buildProviderFunc(configs map[string]*certprovider.BuildableConfig, instanceName, certName string, wantIdentity, wantRoot bool) (certprovider.Provider, error) {
	cfg, ok := configs[instanceName]
	if !ok {
		// Defensive programming. If a resource received from the management
		// server contains a certificate provider instance name that is not
		// found in the bootstrap, the resource is NACKed by the xDS client.
		return nil, fmt.Errorf("certificate provider instance %q not found in bootstrap file", instanceName)
	}
	provider, err := cfg.Build(certprovider.BuildOptions{
		CertName:     certName,
		WantIdentity: wantIdentity,
		WantRoot:     wantRoot,
	})
	if err != nil {
		// This error is not expected since the bootstrap process parses the
		// config and makes sure that it is acceptable to the plugin. Still, it
		// is possible that the plugin parses the config successfully, but its
		// Build() method errors out.
		return nil, fmt.Errorf("failed to get security plugin instance (%+v): %v", cfg, err)
	}
	return provider, nil
}
