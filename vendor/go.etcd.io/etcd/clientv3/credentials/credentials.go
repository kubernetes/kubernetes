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
type transportCredential struct {
	gtc grpccredentials.TransportCredentials
}

func newTransportCredential(cfg *tls.Config) *transportCredential {
	return &transportCredential{
		gtc: grpccredentials.NewTLS(cfg),
	}
}

func (tc *transportCredential) ClientHandshake(ctx context.Context, authority string, rawConn net.Conn) (net.Conn, grpccredentials.AuthInfo, error) {
	// Only overwrite when authority is an IP address!
	// Let's say, a server runs SRV records on "etcd.local" that resolves
	// to "m1.etcd.local", and its SAN field also includes "m1.etcd.local".
	// But what if SAN does not include its resolved IP address (e.g. 127.0.0.1)?
	// Then, the server should only authenticate using its DNS hostname "m1.etcd.local",
	// instead of overwriting it with its IP address.
	// And we do not overwrite "localhost" either. Only overwrite IP addresses!
	if isIP(authority) {
		target := rawConn.RemoteAddr().String()
		if authority != target {
			// When user dials with "grpc.WithDialer", "grpc.DialContext" "cc.parsedTarget"
			// update only happens once. This is problematic, because when TLS is enabled,
			// retries happen through "grpc.WithDialer" with static "cc.parsedTarget" from
			// the initial dial call.
			// If the server authenticates by IP addresses, we want to set a new endpoint as
			// a new authority. Otherwise
			// "transport: authentication handshake failed: x509: certificate is valid for 127.0.0.1, 192.168.121.180, not 192.168.223.156"
			// when the new dial target is "192.168.121.180" whose certificate host name is also "192.168.121.180"
			// but client tries to authenticate with previously set "cc.parsedTarget" field "192.168.223.156"
			authority = target
		}
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
	return &transportCredential{
		gtc: tc.gtc.Clone(),
	}
}

func (tc *transportCredential) OverrideServerName(serverNameOverride string) error {
	return tc.gtc.OverrideServerName(serverNameOverride)
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
