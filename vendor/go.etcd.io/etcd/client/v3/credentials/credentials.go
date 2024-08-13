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

	"go.etcd.io/etcd/api/v3/v3rpc/rpctypes"
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
	return tc.gtc.ClientHandshake(ctx, authority, rawConn)
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
	if authToken == "" {
		return nil, nil
	}
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
