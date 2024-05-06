/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package httpstream

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const (
	HeaderConnection               = "Connection"
	HeaderUpgrade                  = "Upgrade"
	HeaderProtocolVersion          = "X-Stream-Protocol-Version"
	HeaderAcceptedProtocolVersions = "X-Accepted-Stream-Protocol-Versions"
)

// NewStreamHandler defines a function that is called when a new Stream is
// received. If no error is returned, the Stream is accepted; otherwise,
// the stream is rejected. After the reply frame has been sent, replySent is closed.
type NewStreamHandler func(stream Stream, replySent <-chan struct{}) error

// NoOpNewStreamHandler is a stream handler that accepts a new stream and
// performs no other logic.
func NoOpNewStreamHandler(stream Stream, replySent <-chan struct{}) error { return nil }

// Dialer knows how to open a streaming connection to a server.
type Dialer interface {

	// Dial opens a streaming connection to a server using one of the protocols
	// specified (in order of most preferred to least preferred).
	Dial(protocols ...string) (Connection, string, error)
}

// UpgradeRoundTripper is a type of http.RoundTripper that is able to upgrade
// HTTP requests to support multiplexed bidirectional streams. After RoundTrip()
// is invoked, if the upgrade is successful, clients may retrieve the upgraded
// connection by calling UpgradeRoundTripper.Connection().
type UpgradeRoundTripper interface {
	http.RoundTripper
	// NewConnection validates the response and creates a new Connection.
	NewConnection(resp *http.Response) (Connection, error)
}

// ResponseUpgrader knows how to upgrade HTTP requests and responses to
// add streaming support to them.
type ResponseUpgrader interface {
	// UpgradeResponse upgrades an HTTP response to one that supports multiplexed
	// streams. newStreamHandler will be called asynchronously whenever the
	// other end of the upgraded connection creates a new stream.
	UpgradeResponse(w http.ResponseWriter, req *http.Request, newStreamHandler NewStreamHandler) Connection
}

// Connection represents an upgraded HTTP connection.
type Connection interface {
	// CreateStream creates a new Stream with the supplied headers.
	CreateStream(headers http.Header) (Stream, error)
	// Close resets all streams and closes the connection.
	Close() error
	// CloseChan returns a channel that is closed when the underlying connection is closed.
	CloseChan() <-chan bool
	// SetIdleTimeout sets the amount of time the connection may remain idle before
	// it is automatically closed.
	SetIdleTimeout(timeout time.Duration)
	// RemoveStreams can be used to remove a set of streams from the Connection.
	RemoveStreams(streams ...Stream)
}

// Stream represents a bidirectional communications channel that is part of an
// upgraded connection.
type Stream interface {
	io.ReadWriteCloser
	// Reset closes both directions of the stream, indicating that neither client
	// or server can use it any more.
	Reset() error
	// Headers returns the headers used to create the stream.
	Headers() http.Header
	// Identifier returns the stream's ID.
	Identifier() uint32
}

// UpgradeFailureError encapsulates the cause for why the streaming
// upgrade request failed. Implements error interface.
type UpgradeFailureError struct {
	Cause error
}

func (u *UpgradeFailureError) Error() string {
	return fmt.Sprintf("unable to upgrade streaming request: %s", u.Cause)
}

// IsUpgradeFailure returns true if the passed error is (or wrapped error contains)
// the UpgradeFailureError.
func IsUpgradeFailure(err error) bool {
	if err == nil {
		return false
	}
	var upgradeErr *UpgradeFailureError
	return errors.As(err, &upgradeErr)
}

// IsUpgradeRequest returns true if the given request is a connection upgrade request
func IsUpgradeRequest(req *http.Request) bool {
	for _, h := range req.Header[http.CanonicalHeaderKey(HeaderConnection)] {
		if strings.Contains(strings.ToLower(h), strings.ToLower(HeaderUpgrade)) {
			return true
		}
	}
	return false
}

func negotiateProtocol(clientProtocols, serverProtocols []string) string {
	for i := range clientProtocols {
		for j := range serverProtocols {
			if clientProtocols[i] == serverProtocols[j] {
				return clientProtocols[i]
			}
		}
	}
	return ""
}

func commaSeparatedHeaderValues(header []string) []string {
	var parsedClientProtocols []string
	for i := range header {
		for _, clientProtocol := range strings.Split(header[i], ",") {
			if proto := strings.Trim(clientProtocol, " "); len(proto) > 0 {
				parsedClientProtocols = append(parsedClientProtocols, proto)
			}
		}
	}
	return parsedClientProtocols
}

// Handshake performs a subprotocol negotiation. If the client did request a
// subprotocol, Handshake will select the first common value found in
// serverProtocols. If a match is found, Handshake adds a response header
// indicating the chosen subprotocol. If no match is found, HTTP forbidden is
// returned, along with a response header containing the list of protocols the
// server can accept.
func Handshake(req *http.Request, w http.ResponseWriter, serverProtocols []string) (string, error) {
	clientProtocols := commaSeparatedHeaderValues(req.Header[http.CanonicalHeaderKey(HeaderProtocolVersion)])
	if len(clientProtocols) == 0 {
		return "", fmt.Errorf("unable to upgrade: %s is required", HeaderProtocolVersion)
	}

	if len(serverProtocols) == 0 {
		panic(fmt.Errorf("unable to upgrade: serverProtocols is required"))
	}

	negotiatedProtocol := negotiateProtocol(clientProtocols, serverProtocols)
	if len(negotiatedProtocol) == 0 {
		for i := range serverProtocols {
			w.Header().Add(HeaderAcceptedProtocolVersions, serverProtocols[i])
		}
		err := fmt.Errorf("unable to upgrade: unable to negotiate protocol: client supports %v, server accepts %v", clientProtocols, serverProtocols)
		http.Error(w, err.Error(), http.StatusForbidden)
		return "", err
	}

	w.Header().Add(HeaderProtocolVersion, negotiatedProtocol)
	return negotiatedProtocol, nil
}
