// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package websocket

import (
	"bufio"
	"context"
	"io"
	"net"
	"net/http"
	"net/url"
	"time"
)

// DialError is an error that occurs while dialling a websocket server.
type DialError struct {
	*Config
	Err error
}

func (e *DialError) Error() string {
	return "websocket.Dial " + e.Config.Location.String() + ": " + e.Err.Error()
}

// NewConfig creates a new WebSocket config for client connection.
func NewConfig(server, origin string) (config *Config, err error) {
	config = new(Config)
	config.Version = ProtocolVersionHybi13
	config.Location, err = url.ParseRequestURI(server)
	if err != nil {
		return
	}
	config.Origin, err = url.ParseRequestURI(origin)
	if err != nil {
		return
	}
	config.Header = http.Header(make(map[string][]string))
	return
}

// NewClient creates a new WebSocket client connection over rwc.
func NewClient(config *Config, rwc io.ReadWriteCloser) (ws *Conn, err error) {
	br := bufio.NewReader(rwc)
	bw := bufio.NewWriter(rwc)
	err = hybiClientHandshake(config, br, bw)
	if err != nil {
		return
	}
	buf := bufio.NewReadWriter(br, bw)
	ws = newHybiClientConn(config, buf, rwc)
	return
}

// Dial opens a new client connection to a WebSocket.
func Dial(url_, protocol, origin string) (ws *Conn, err error) {
	config, err := NewConfig(url_, origin)
	if err != nil {
		return nil, err
	}
	if protocol != "" {
		config.Protocol = []string{protocol}
	}
	return DialConfig(config)
}

var portMap = map[string]string{
	"ws":  "80",
	"wss": "443",
}

func parseAuthority(location *url.URL) string {
	if _, ok := portMap[location.Scheme]; ok {
		if _, _, err := net.SplitHostPort(location.Host); err != nil {
			return net.JoinHostPort(location.Host, portMap[location.Scheme])
		}
	}
	return location.Host
}

// DialConfig opens a new client connection to a WebSocket with a config.
func DialConfig(config *Config) (ws *Conn, err error) {
	return config.DialContext(context.Background())
}

// DialContext opens a new client connection to a WebSocket, with context support for timeouts/cancellation.
func (config *Config) DialContext(ctx context.Context) (*Conn, error) {
	if config.Location == nil {
		return nil, &DialError{config, ErrBadWebSocketLocation}
	}
	if config.Origin == nil {
		return nil, &DialError{config, ErrBadWebSocketOrigin}
	}

	dialer := config.Dialer
	if dialer == nil {
		dialer = &net.Dialer{}
	}

	client, err := dialWithDialer(ctx, dialer, config)
	if err != nil {
		return nil, &DialError{config, err}
	}

	// Cleanup the connection if we fail to create the websocket successfully
	success := false
	defer func() {
		if !success {
			_ = client.Close()
		}
	}()

	var ws *Conn
	var wsErr error
	doneConnecting := make(chan struct{})
	go func() {
		defer close(doneConnecting)
		ws, err = NewClient(config, client)
		if err != nil {
			wsErr = &DialError{config, err}
		}
	}()

	// The websocket.NewClient() function can block indefinitely, make sure that we
	// respect the deadlines specified by the context.
	select {
	case <-ctx.Done():
		// Force the pending operations to fail, terminating the pending connection attempt
		_ = client.SetDeadline(time.Now())
		<-doneConnecting // Wait for the goroutine that tries to establish the connection to finish
		return nil, &DialError{config, ctx.Err()}
	case <-doneConnecting:
		if wsErr == nil {
			success = true // Disarm the deferred connection cleanup
		}
		return ws, wsErr
	}
}
