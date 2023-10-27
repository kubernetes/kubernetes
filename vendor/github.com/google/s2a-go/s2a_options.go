/*
 *
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package s2a

import (
	"context"
	"crypto/tls"
	"errors"
	"sync"

	"github.com/google/s2a-go/fallback"
	"github.com/google/s2a-go/stream"

	s2apb "github.com/google/s2a-go/internal/proto/common_go_proto"
)

// Identity is the interface for S2A identities.
type Identity interface {
	// Name returns the name of the identity.
	Name() string
}

type spiffeID struct {
	spiffeID string
}

func (s *spiffeID) Name() string { return s.spiffeID }

// NewSpiffeID creates a SPIFFE ID from id.
func NewSpiffeID(id string) Identity {
	return &spiffeID{spiffeID: id}
}

type hostname struct {
	hostname string
}

func (h *hostname) Name() string { return h.hostname }

// NewHostname creates a hostname from name.
func NewHostname(name string) Identity {
	return &hostname{hostname: name}
}

type uid struct {
	uid string
}

func (h *uid) Name() string { return h.uid }

// NewUID creates a UID from name.
func NewUID(name string) Identity {
	return &uid{uid: name}
}

// VerificationModeType specifies the mode that S2A must use to verify the peer
// certificate chain.
type VerificationModeType int

// Three types of verification modes.
const (
	Unspecified = iota
	ConnectToGoogle
	Spiffe
)

// ClientOptions contains the client-side options used to establish a secure
// channel using the S2A handshaker service.
type ClientOptions struct {
	// TargetIdentities contains a list of allowed server identities. One of the
	// target identities should match the peer identity in the handshake
	// result; otherwise, the handshake fails.
	TargetIdentities []Identity
	// LocalIdentity is the local identity of the client application. If none is
	// provided, then the S2A will choose the default identity, if one exists.
	LocalIdentity Identity
	// S2AAddress is the address of the S2A.
	S2AAddress string
	// EnsureProcessSessionTickets waits for all session tickets to be sent to
	// S2A before a process completes.
	//
	// This functionality is crucial for processes that complete very soon after
	// using S2A to establish a TLS connection, but it can be ignored for longer
	// lived processes.
	//
	// Usage example:
	//   func main() {
	//     var ensureProcessSessionTickets sync.WaitGroup
	//     clientOpts := &s2a.ClientOptions{
	//       EnsureProcessSessionTickets: &ensureProcessSessionTickets,
	//       // Set other members.
	//     }
	//     creds, _ := s2a.NewClientCreds(clientOpts)
	//     conn, _ := grpc.Dial(serverAddr, grpc.WithTransportCredentials(creds))
	//     defer conn.Close()
	//
	//     // Make RPC call.
	//
	//     // The process terminates right after the RPC call ends.
	//     // ensureProcessSessionTickets can be used to ensure resumption
	//     // tickets are fully processed. If the process is long-lived, using
	//     // ensureProcessSessionTickets is not necessary.
	//     ensureProcessSessionTickets.Wait()
	//   }
	EnsureProcessSessionTickets *sync.WaitGroup
	// If true, enables the use of legacy S2Av1.
	EnableLegacyMode bool
	// VerificationMode specifies the mode that S2A must use to verify the
	// peer certificate chain.
	VerificationMode VerificationModeType

	// Optional fallback after dialing with S2A fails.
	FallbackOpts *FallbackOptions

	// Generates an S2AStream interface for talking to the S2A server.
	getS2AStream func(ctx context.Context, s2av2Address string) (stream.S2AStream, error)

	// Serialized user specified policy for server authorization.
	serverAuthorizationPolicy []byte
}

// FallbackOptions prescribes the fallback logic that should be taken if the application fails to connect with S2A.
type FallbackOptions struct {
	// FallbackClientHandshakeFunc is used to specify fallback behavior when calling s2a.NewClientCreds().
	// It will be called by ClientHandshake function, after handshake with S2A fails.
	// s2a.NewClientCreds() ignores the other FallbackDialer field.
	FallbackClientHandshakeFunc fallback.ClientHandshake

	// FallbackDialer is used to specify fallback behavior when calling s2a.NewS2aDialTLSContextFunc().
	// It passes in a custom fallback dialer and server address to use after dialing with S2A fails.
	// s2a.NewS2aDialTLSContextFunc() ignores the other FallbackClientHandshakeFunc field.
	FallbackDialer *FallbackDialer
}

// FallbackDialer contains a fallback tls.Dialer and a server address to connect to.
type FallbackDialer struct {
	// Dialer specifies a fallback tls.Dialer.
	Dialer *tls.Dialer
	// ServerAddr is used by Dialer to establish fallback connection.
	ServerAddr string
}

// DefaultClientOptions returns the default client options.
func DefaultClientOptions(s2aAddress string) *ClientOptions {
	return &ClientOptions{
		S2AAddress:       s2aAddress,
		VerificationMode: ConnectToGoogle,
	}
}

// ServerOptions contains the server-side options used to establish a secure
// channel using the S2A handshaker service.
type ServerOptions struct {
	// LocalIdentities is the list of local identities that may be assumed by
	// the server. If no local identity is specified, then the S2A chooses a
	// default local identity, if one exists.
	LocalIdentities []Identity
	// S2AAddress is the address of the S2A.
	S2AAddress string
	// If true, enables the use of legacy S2Av1.
	EnableLegacyMode bool
	// VerificationMode specifies the mode that S2A must use to verify the
	// peer certificate chain.
	VerificationMode VerificationModeType

	// Generates an S2AStream interface for talking to the S2A server.
	getS2AStream func(ctx context.Context, s2av2Address string) (stream.S2AStream, error)
}

// DefaultServerOptions returns the default server options.
func DefaultServerOptions(s2aAddress string) *ServerOptions {
	return &ServerOptions{
		S2AAddress:       s2aAddress,
		VerificationMode: ConnectToGoogle,
	}
}

func toProtoIdentity(identity Identity) (*s2apb.Identity, error) {
	if identity == nil {
		return nil, nil
	}
	switch id := identity.(type) {
	case *spiffeID:
		return &s2apb.Identity{IdentityOneof: &s2apb.Identity_SpiffeId{SpiffeId: id.Name()}}, nil
	case *hostname:
		return &s2apb.Identity{IdentityOneof: &s2apb.Identity_Hostname{Hostname: id.Name()}}, nil
	case *uid:
		return &s2apb.Identity{IdentityOneof: &s2apb.Identity_Uid{Uid: id.Name()}}, nil
	default:
		return nil, errors.New("unrecognized identity type")
	}
}
