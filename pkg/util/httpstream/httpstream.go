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
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
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
}

// Stream represents a bidirectional communications channel that is part of an
// upgraded connection.
//
// NOTE: in contrast to Connection, here Close() closes only the sending half of the stream
//       and Reset() closes both.
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

// Handshake performs a subprotocol negotiation. If the client did request a
// subprotocol, Handshake will select the first common value found in
// serverProtocols. If a match is found, Handshake adds a response header
// indicating the chosen subprotocol. If no match is found, HTTP forbidden is
// returned, along with a response header containing the list of protocols the
// server can accept.
func Handshake(req *http.Request, w http.ResponseWriter, serverProtocols []string) (string, error) {
	clientProtocols := req.Header[http.CanonicalHeaderKey(HeaderProtocolVersion)]
	if len(clientProtocols) == 0 {
		// Kube 1.0 clients didn't support subprotocol negotiation.
		// TODO require clientProtocols once Kube 1.0 is no longer supported
		return "", nil
	}

	if len(serverProtocols) == 0 {
		// Kube 1.0 servers didn't support subprotocol negotiation. This is mainly for testing.
		// TODO require serverProtocols once Kube 1.0 is no longer supported
		return "", nil
	}

	negotiatedProtocol := negotiateProtocol(clientProtocols, serverProtocols)
	if len(negotiatedProtocol) == 0 {
		w.WriteHeader(http.StatusForbidden)
		for i := range serverProtocols {
			w.Header().Add(HeaderAcceptedProtocolVersions, serverProtocols[i])
		}
		fmt.Fprintf(w, "unable to upgrade: unable to negotiate protocol: client supports %v, server accepts %v", clientProtocols, serverProtocols)
		return "", fmt.Errorf("unable to upgrade: unable to negotiate protocol: client supports %v, server supports %v", clientProtocols, serverProtocols)
	}

	w.Header().Add(HeaderProtocolVersion, negotiatedProtocol)
	return negotiatedProtocol, nil
}

type tcpReadCloser struct {
	*net.TCPConn
}

type tcpWriteCloser struct {
	*net.TCPConn
}

func (rc tcpReadCloser) Close() error {
	return rc.TCPConn.CloseRead()
}

func (wc tcpWriteCloser) Close() error {
	return wc.TCPConn.CloseWrite()
}

type streadReadCloser struct {
	Stream
}

type streamWriteCloser struct {
	Stream
}

func (rc streadReadCloser) Close() error {
	return rc.Stream.Close()
}

func (wc streamWriteCloser) Close() error {
	return wc.Stream.Reset() // this closes both sides. Best we can do for a Stream.
}

// SplitTCPConn splits a TCP connection into a WriteCloser and ReadCloser. Closing one of them turns an open
// connection into an half-open one.
func SplitTCPConn(conn *net.TCPConn) (io.WriteCloser, io.ReadCloser) {
	return tcpWriteCloser{conn}, tcpReadCloser{conn}
}

// SplitTCPConn splits a httpstream.Stream into a WriteCloser and ReadCloser. Closing the former closes the
// the write half of the stream, closing the later closes both halfs.
func SplitStream(stream Stream) (io.WriteCloser, io.ReadCloser) {
	return streadReadCloser{stream}, streamWriteCloser{stream}
}

type tee struct {
	io.ReadCloser
	id string
}

func (t tee) Read(p []byte) (n int, err error) {
	n, err = t.ReadCloser.Read(p)
	if err != nil {
		fmt.Printf("portforward %s read error: %v\n", t.id, err)
	} else {
		fmt.Printf("portforward %s read %d bytes: %q\n", t.id, n, p[0:n])
	}
	return
}

// CopyBothWays copies data from conn to dataStream and the other way around. An EOF on either side is passed along
// as a one-sided Close call. On error the connection state is undefined and must be cleaned up by the caller.
// If no error occurs, both connections are kept half-open.
func CopyBothWays(remoteIn io.WriteCloser, remoteOut io.ReadCloser, localIn io.WriteCloser, localOut io.ReadCloser, okishErrorPatterns []string) error {
	localError := make(chan error, 2)
	copiesDone := make(chan struct{})

	copiesGroup := sync.WaitGroup{}
	copiesGroup.Add(2)

	// Copy from the remote side to the local port.
	go func() {
		defer copiesGroup.Done()
		defer remoteIn.Close()
		if _, err := io.Copy(remoteIn, tee{ReadCloser: localOut, id: "remote"}); err != nil {
			for _, s := range okishErrorPatterns {
				if strings.Contains(err.Error(), s) {
					return
				}
			}
			localError <- err
		}
	}()

	// Copy from the local port to the remote side.
	go func() {
		defer copiesGroup.Done()
		defer localIn.Close()
		if _, err := io.Copy(localIn, tee{ReadCloser: remoteOut, id: "local"}); err != nil {
			for _, s := range okishErrorPatterns {
				if strings.Contains(err.Error(), s) {
					return
				}
			}
			localError <- err
		}
	}()

	go func() {
		defer close(copiesDone)
		copiesGroup.Wait()
	}()

	// wait for either both copies finish or an error occurs
	select {
	case <-copiesDone:
	case err := <-localError: // catch the first error, potentially ignore the second
		return err
	}

	return nil
}
