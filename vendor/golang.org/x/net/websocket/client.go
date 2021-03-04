// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package websocket

import (
	"bufio"
	"io"
	"net"
	"net/http"
	"net/url"
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
	var client net.Conn
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
	client, err = dialWithDialer(dialer, config)
	if err != nil {
		goto Error
	}
	ws, err = NewClient(config, client)
	if err != nil {
		client.Close()
		goto Error
	}
	return

Error:
	return nil, &DialError{config, err}
}
