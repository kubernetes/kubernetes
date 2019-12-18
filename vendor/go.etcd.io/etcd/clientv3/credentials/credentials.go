// Copyright 2019 The etcd Authors
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

// Package credentials implements gRPC credential interface with etcd specific logic.
// e.g., client handshake with custom authority parameter
package credentials

import (
	"context"
	"crypto/tls"
	"net"
	"sync"

	"go.etcd.io/etcd/clientv3/balancer/resolver/endpoint"
	"go.etcd.io/etcd/etcdserver/api/v3rpc/rpctypes"
	grpccredentials "google.golang.org/grpc/credentials"
)

// Config defines gRPC credential configuration.
type Config struct {
	TLSConfig *tls.Config
}

// Bundle defines gRPC credential interface.
type Bundle interface {
	grpccredentials.Bundle
	UpdateAuthToken(token string)
}

// NewBundle constructs a new gRPC credential bundle.
func NewBundle(cfg Config) Bundle {
	return &bundle{
		tc: newTransportCredential(cfg.TLSConfig),
		rc: newPerRPCCredential(),
	}
}

// bundle implements "grpccredentials.Bundle" interface.
type bundle struct {
	tc *transportCredential
	rc *perRPCCredential
}

func (b *bundle) TransportCredentials() grpccredentials.TransportCredentials {
	return b.tc
}

func (b *bundle) PerRPCCredentials() grpccredentials.PerRPCCredentials {
	return b.rc
}

func (b *bundle) NewWithMode(mode string) (grpccredentials.Bundle, error) {
	// no-op
	return nil, nil
}

// transportCredential implements "grpccredentials.TransportCredentials" interface.
// transportCredential wraps TransportCredentials to track which
// addresses are dialed for which endpoints, and then sets the authority when checking the endpoint's cert to the
// hostname or IP of the dialed endpoint.
// This is a workaround of a gRPC load balancer issue. gRPC uses the dialed target's service name as the authority when
// checking all endpoint certs, which does not work for etcd servers using their hostname or IP as the Subject Alternative Name
// in their TLS certs.
// To enable, include both WithTransportCredentials(creds) and WithContextDialer(creds.Dialer)
// when dialing.
type transportCredential struct {
	gtc grpccredentials.TransportCredentials
	mu  sync.Mutex
	// addrToEndpoint maps from the connection addresses that are dialed to the hostname or IP of the
	// endpoint provided to the dialer when dialing
	addrToEndpoint map[string]string
}

func newTransportCredential(cfg *tls.Config) *transportCredential {
	return &transportCredential{
		gtc:            grpccredentials.NewTLS(cfg),
		addrToEndpoint: map[string]string{},
	}
}

func (tc *transportCredential) ClientHandshake(ctx context.Context, authority string, rawConn net.Conn) (net.Conn, grpccredentials.AuthInfo, error) {
	// Set the authority when checking the endpoint's cert to the hostname or IP of the dialed endpoint
	tc.mu.Lock()
	dialEp, ok := tc.addrToEndpoint[rawConn.RemoteAddr().String()]
	tc.mu.Unlock()
	if ok {
		_, host, _ := endpoint.ParseEndpoint(dialEp)
		authority = host
	}
	return tc.gtc.ClientHandshake(ctx, authority, rawConn)
}

// return true if given string is an IP.
func isIP(ep string) bool {
	return net.ParseIP(ep) != nil
}

func (tc *transportCredential) ServerHandshake(rawConn net.Conn) (net.Conn, grpccredentials.AuthInfo, error) {
	return tc.gtc.ServerHandshake(rawConn)
}

func (tc *transportCredential) Info() grpccredentials.ProtocolInfo {
	return tc.gtc.Info()
}

func (tc *transportCredential) Clone() grpccredentials.TransportCredentials {
	copy := map[string]string{}
	tc.mu.Lock()
	for k, v := range tc.addrToEndpoint {
		copy[k] = v
	}
	tc.mu.Unlock()
	return &transportCredential{
		gtc:            tc.gtc.Clone(),
		addrToEndpoint: copy,
	}
}

func (tc *transportCredential) OverrideServerName(serverNameOverride string) error {
	return tc.gtc.OverrideServerName(serverNameOverride)
}

func (tc *transportCredential) Dialer(ctx context.Context, dialEp string) (net.Conn, error) {
	// Keep track of which addresses are dialed for which endpoints
	conn, err := endpoint.Dialer(ctx, dialEp)
	if conn != nil {
		tc.mu.Lock()
		tc.addrToEndpoint[conn.RemoteAddr().String()] = dialEp
		tc.mu.Unlock()
	}
	return conn, err
}

// perRPCCredential implements "grpccredentials.PerRPCCredentials" interface.
type perRPCCredential struct {
	authToken   string
	authTokenMu sync.RWMutex
}

func newPerRPCCredential() *perRPCCredential { return &perRPCCredential{} }

func (rc *perRPCCredential) RequireTransportSecurity() bool { return false }

func (rc *perRPCCredential) GetRequestMetadata(ctx context.Context, s ...string) (map[string]string, error) {
	rc.authTokenMu.RLock()
	authToken := rc.authToken
	rc.authTokenMu.RUnlock()
	return map[string]string{rpctypes.TokenFieldNameGRPC: authToken}, nil
}

func (b *bundle) UpdateAuthToken(token string) {
	if b.rc == nil {
		return
	}
	b.rc.UpdateAuthToken(token)
}

func (rc *perRPCCredential) UpdateAuthToken(token string) {
	rc.authTokenMu.Lock()
	rc.authToken = token
	rc.authTokenMu.Unlock()
}
