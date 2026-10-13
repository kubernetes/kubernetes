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

package spdy

import (
	"crypto/tls"
	"net"
	"net/http"
	"net/url"
	"time"

	apihttpstream "k8s.io/apimachinery/pkg/util/httpstream"
	streamhttp "k8s.io/streaming/pkg/httpstream"
	streamspdy "k8s.io/streaming/pkg/httpstream/spdy"
)

const HeaderSpdy31 = streamspdy.HeaderSpdy31

// SpdyRoundTripper is a compatibility wrapper around the streaming module's
// SPDY round tripper.
type SpdyRoundTripper struct {
	delegate *streamspdy.SpdyRoundTripper
}

func NewRoundTripper(tlsConfig *tls.Config) (*SpdyRoundTripper, error) {
	delegate, err := streamspdy.NewRoundTripper(tlsConfig)
	if err != nil {
		return nil, err
	}
	return &SpdyRoundTripper{delegate: delegate}, nil
}

func NewRoundTripperWithProxy(tlsConfig *tls.Config, proxier func(*http.Request) (*url.URL, error)) (*SpdyRoundTripper, error) {
	delegate, err := streamspdy.NewRoundTripperWithProxy(tlsConfig, proxier)
	if err != nil {
		return nil, err
	}
	return &SpdyRoundTripper{delegate: delegate}, nil
}

// RoundTripperConfig is a set of options for an SpdyRoundTripper.
type RoundTripperConfig struct {
	// TLS configuration used by the round tripper if UpgradeTransport not present.
	TLS *tls.Config
	// Proxier is a proxy function invoked on each request. Optional.
	Proxier func(*http.Request) (*url.URL, error)
	// PingPeriod is a period for sending SPDY Pings on the connection.
	// Optional.
	PingPeriod time.Duration
	// UpgradeTransport is a subtitute transport used for dialing. If set,
	// this field will be used instead of "TLS" and "Proxier" for connection creation.
	// Optional.
	UpgradeTransport http.RoundTripper
}

func NewRoundTripperWithConfig(cfg RoundTripperConfig) (*SpdyRoundTripper, error) {
	delegate, err := streamspdy.NewRoundTripperWithConfig(streamspdy.RoundTripperConfig{
		TLS:              cfg.TLS,
		Proxier:          cfg.Proxier,
		PingPeriod:       cfg.PingPeriod,
		UpgradeTransport: cfg.UpgradeTransport,
	})
	if err != nil {
		return nil, err
	}
	return &SpdyRoundTripper{delegate: delegate}, nil
}

// TLSClientConfig implements pkg/util/net.TLSClientConfigHolder for proper TLS checking during
// proxying with a spdy roundtripper.
func (s *SpdyRoundTripper) TLSClientConfig() *tls.Config {
	return s.delegate.TLSClientConfig()
}

// Dial opens a network connection for an upgrade request.
func (s *SpdyRoundTripper) Dial(req *http.Request) (net.Conn, error) {
	return s.delegate.Dial(req)
}

// RoundTrip executes a request and upgrades the connection.
func (s *SpdyRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	return s.delegate.RoundTrip(req)
}

// NewConnection validates a server upgrade response and prepares the transport.
func (s *SpdyRoundTripper) NewConnection(resp *http.Response) (apihttpstream.Connection, error) {
	conn, err := s.delegate.NewConnection(resp)
	if err != nil {
		return nil, err
	}
	return wrapConnection(conn), nil
}

type responseUpgraderAdapter struct {
	delegate streamhttp.ResponseUpgrader
}

func (r *responseUpgraderAdapter) UpgradeResponse(w http.ResponseWriter, req *http.Request, newStreamHandler apihttpstream.NewStreamHandler) apihttpstream.Connection {
	conn := r.delegate.UpgradeResponse(w, req, wrapNewStreamHandler(newStreamHandler))
	return wrapConnection(conn)
}

func NewResponseUpgrader() apihttpstream.ResponseUpgrader {
	return &responseUpgraderAdapter{delegate: streamspdy.NewResponseUpgrader()}
}

func NewResponseUpgraderWithPings(pingPeriod time.Duration) apihttpstream.ResponseUpgrader {
	return &responseUpgraderAdapter{delegate: streamspdy.NewResponseUpgraderWithPings(pingPeriod)}
}

func NewClientConnection(conn net.Conn) (apihttpstream.Connection, error) {
	c, err := streamspdy.NewClientConnection(conn)
	if err != nil {
		return nil, err
	}
	return wrapConnection(c), nil
}

func NewClientConnectionWithPings(conn net.Conn, pingPeriod time.Duration) (apihttpstream.Connection, error) {
	c, err := streamspdy.NewClientConnectionWithPings(conn, pingPeriod)
	if err != nil {
		return nil, err
	}
	return wrapConnection(c), nil
}

func NewServerConnection(conn net.Conn, newStreamHandler apihttpstream.NewStreamHandler) (apihttpstream.Connection, error) {
	c, err := streamspdy.NewServerConnection(conn, wrapNewStreamHandler(newStreamHandler))
	if err != nil {
		return nil, err
	}
	return wrapConnection(c), nil
}

func NewServerConnectionWithPings(conn net.Conn, newStreamHandler apihttpstream.NewStreamHandler, pingPeriod time.Duration) (apihttpstream.Connection, error) {
	c, err := streamspdy.NewServerConnectionWithPings(conn, wrapNewStreamHandler(newStreamHandler), pingPeriod)
	if err != nil {
		return nil, err
	}
	return wrapConnection(c), nil
}

type streamAdapter struct {
	delegate streamhttp.Stream
}

func (s *streamAdapter) Read(p []byte) (int, error) {
	return s.delegate.Read(p)
}

func (s *streamAdapter) Write(p []byte) (int, error) {
	return s.delegate.Write(p)
}

func (s *streamAdapter) Close() error {
	return s.delegate.Close()
}

func (s *streamAdapter) Reset() error {
	return s.delegate.Reset()
}

func (s *streamAdapter) Headers() http.Header {
	return s.delegate.Headers()
}

func (s *streamAdapter) Identifier() uint32 {
	return s.delegate.Identifier()
}

type connectionAdapter struct {
	delegate streamhttp.Connection
}

func (c *connectionAdapter) CreateStream(headers http.Header) (apihttpstream.Stream, error) {
	stream, err := c.delegate.CreateStream(headers)
	if err != nil {
		return nil, err
	}
	return &streamAdapter{delegate: stream}, nil
}

func (c *connectionAdapter) Close() error {
	return c.delegate.Close()
}

func (c *connectionAdapter) CloseChan() <-chan bool {
	return c.delegate.CloseChan()
}

func (c *connectionAdapter) SetIdleTimeout(timeout time.Duration) {
	c.delegate.SetIdleTimeout(timeout)
}

func (c *connectionAdapter) RemoveStreams(streams ...apihttpstream.Stream) {
	streamingStreams := make([]streamhttp.Stream, 0, len(streams))
	for _, stream := range streams {
		if stream == nil {
			continue
		}
		if s, ok := stream.(streamhttp.Stream); ok {
			streamingStreams = append(streamingStreams, s)
		}
	}
	c.delegate.RemoveStreams(streamingStreams...)
}

func wrapConnection(conn streamhttp.Connection) apihttpstream.Connection {
	if conn == nil {
		return nil
	}
	return &connectionAdapter{delegate: conn}
}

func wrapNewStreamHandler(newStreamHandler apihttpstream.NewStreamHandler) streamhttp.NewStreamHandler {
	if newStreamHandler == nil {
		return nil
	}
	return func(stream streamhttp.Stream, replySent <-chan struct{}) error {
		return newStreamHandler(&streamAdapter{delegate: stream}, replySent)
	}
}
