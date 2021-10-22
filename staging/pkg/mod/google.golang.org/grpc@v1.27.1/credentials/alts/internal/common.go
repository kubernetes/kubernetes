/*
 *
 * Copyright 2018 gRPC authors.
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

//go:generate ./regenerate.sh

// Package internal contains common core functionality for ALTS.
package internal

import (
	"context"
	"net"

	"google.golang.org/grpc/credentials"
)

const (
	// ClientSide identifies the client in this communication.
	ClientSide Side = iota
	// ServerSide identifies the server in this communication.
	ServerSide
)

// PeerNotRespondingError is returned when a peer server is not responding
// after a channel has been established. It is treated as a temporary connection
// error and re-connection to the server should be attempted.
var PeerNotRespondingError = &peerNotRespondingError{}

// Side identifies the party's role: client or server.
type Side int

type peerNotRespondingError struct{}

// Return an error message for the purpose of logging.
func (e *peerNotRespondingError) Error() string {
	return "peer server is not responding and re-connection should be attempted."
}

// Temporary indicates if this connection error is temporary or fatal.
func (e *peerNotRespondingError) Temporary() bool {
	return true
}

// Handshaker defines a ALTS handshaker interface.
type Handshaker interface {
	// ClientHandshake starts and completes a client-side handshaking and
	// returns a secure connection and corresponding auth information.
	ClientHandshake(ctx context.Context) (net.Conn, credentials.AuthInfo, error)
	// ServerHandshake starts and completes a server-side handshaking and
	// returns a secure connection and corresponding auth information.
	ServerHandshake(ctx context.Context) (net.Conn, credentials.AuthInfo, error)
	// Close terminates the Handshaker. It should be called when the caller
	// obtains the secure connection.
	Close()
}
