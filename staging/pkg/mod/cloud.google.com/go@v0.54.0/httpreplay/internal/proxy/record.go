// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package proxy provides a record/replay HTTP proxy. It is designed to support
// both an in-memory API (cloud.google.com/go/httpreplay) and a standalone server
// (cloud.google.com/go/httpreplay/cmd/httpr).
package proxy

// See github.com/google/martian/cmd/proxy/main.go for the origin of much of this.

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/google/martian"
	"github.com/google/martian/fifo"
	"github.com/google/martian/httpspec"
	"github.com/google/martian/martianlog"
	"github.com/google/martian/mitm"
)

// A Proxy is an HTTP proxy that supports recording or replaying requests.
type Proxy struct {
	// The certificate that the proxy uses to participate in TLS.
	CACert *x509.Certificate

	// The URL of the proxy.
	URL *url.URL

	// Initial state of the client.
	Initial []byte

	mproxy        *martian.Proxy
	filename      string          // for log
	logger        *Logger         // for recording only
	ignoreHeaders map[string]bool // headers the user has asked to ignore
}

// ForRecording returns a Proxy configured to record.
func ForRecording(filename string, port int) (*Proxy, error) {
	p, err := newProxy(filename)
	if err != nil {
		return nil, err
	}

	// Construct a group that performs the standard proxy stack of request/response
	// modifications.
	stack, _ := httpspec.NewStack("httpr") // second arg is an internal group that we don't need
	p.mproxy.SetRequestModifier(stack)
	p.mproxy.SetResponseModifier(stack)

	// Make a group for logging requests and responses.
	logGroup := fifo.NewGroup()
	skipAuth := skipLoggingByHost("accounts.google.com")
	logGroup.AddRequestModifier(skipAuth)
	logGroup.AddResponseModifier(skipAuth)
	p.logger = newLogger()
	logGroup.AddRequestModifier(p.logger)
	logGroup.AddResponseModifier(p.logger)

	stack.AddRequestModifier(logGroup)
	stack.AddResponseModifier(logGroup)

	// Ordinary debug logging.
	logger := martianlog.NewLogger()
	logger.SetDecode(true)
	stack.AddRequestModifier(logger)
	stack.AddResponseModifier(logger)

	if err := p.start(port); err != nil {
		return nil, err
	}
	return p, nil
}

var (
	configOnce sync.Once
	cert       *x509.Certificate
	config     *mitm.Config
	configErr  error
)

func newProxy(filename string) (*Proxy, error) {
	configOnce.Do(func() {
		// Set up a man-in-the-middle configuration with a CA certificate so the proxy can
		// participate in TLS.
		x509c, priv, err := mitm.NewAuthority("cloud.google.com/go/httpreplay", "HTTPReplay Authority", 100*time.Hour)
		if err != nil {
			configErr = err
			return
		}
		cert = x509c
		config, configErr = mitm.NewConfig(x509c, priv)
		if config != nil {
			config.SetValidity(100 * time.Hour)
			config.SetOrganization("cloud.google.com/go/httpreplay")
			config.SkipTLSVerify(false)
		}
	})
	if configErr != nil {
		return nil, configErr
	}
	mproxy := martian.NewProxy()
	mproxy.SetMITM(config)
	return &Proxy{
		mproxy:        mproxy,
		CACert:        cert,
		filename:      filename,
		ignoreHeaders: map[string]bool{},
	}, nil
}

func (p *Proxy) start(port int) error {
	l, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return err
	}
	p.URL = &url.URL{Scheme: "http", Host: l.Addr().String()}
	go p.mproxy.Serve(l)
	return nil
}

// Transport returns an http.Transport for clients who want to talk to the proxy.
func (p *Proxy) Transport() *http.Transport {
	caCertPool := x509.NewCertPool()
	caCertPool.AddCert(p.CACert)
	return &http.Transport{
		TLSClientConfig: &tls.Config{RootCAs: caCertPool},
		Proxy:           func(*http.Request) (*url.URL, error) { return p.URL, nil },
	}
}

// RemoveRequestHeaders will remove request headers matching patterns from the log,
// and skip matching them. Pattern is taken literally except for *, which matches any
// sequence of characters.
//
// This only needs to be called during recording; the patterns will be saved to the
// log for replay.
func (p *Proxy) RemoveRequestHeaders(patterns []string) {
	for _, pat := range patterns {
		p.logger.log.Converter.registerRemoveRequestHeaders(pat)
	}
}

// ClearHeaders will replace matching headers with CLEARED.
//
// This only needs to be called during recording; the patterns will be saved to the
// log for replay.
func (p *Proxy) ClearHeaders(patterns []string) {
	for _, pat := range patterns {
		p.logger.log.Converter.registerClearHeaders(pat)
	}
}

// RemoveQueryParams will remove query parameters matching patterns from the request
// URL before logging, and skip matching them. Pattern is taken literally except for
// *, which matches any sequence of characters.
//
// This only needs to be called during recording; the patterns will be saved to the
// log for replay.
func (p *Proxy) RemoveQueryParams(patterns []string) {
	for _, pat := range patterns {
		p.logger.log.Converter.registerRemoveParams(pat)
	}
}

// ClearQueryParams will replace matching query params in the request URL with CLEARED.
//
// This only needs to be called during recording; the patterns will be saved to the
// log for replay.
func (p *Proxy) ClearQueryParams(patterns []string) {
	for _, pat := range patterns {
		p.logger.log.Converter.registerClearParams(pat)
	}
}

// IgnoreHeader will cause h to be ignored during matching on replay.
// Deprecated: use RemoveRequestHeaders instead.
func (p *Proxy) IgnoreHeader(h string) {
	p.ignoreHeaders[http.CanonicalHeaderKey(h)] = true
}

// Close closes the proxy. If the proxy is recording, it also writes the log.
func (p *Proxy) Close() error {
	p.mproxy.Close()
	if p.logger != nil {
		return p.writeLog()
	}
	return nil
}

func (p *Proxy) writeLog() error {
	lg := p.logger.Extract()
	lg.Initial = p.Initial
	bytes, err := json.MarshalIndent(lg, "", "  ")
	if err != nil {
		return err
	}
	return ioutil.WriteFile(p.filename, bytes, 0600) // only accessible by owner
}

// skipLoggingByHost disables logging for traffic to a particular host.
type skipLoggingByHost string

func (s skipLoggingByHost) ModifyRequest(req *http.Request) error {
	if strings.HasPrefix(req.Host, string(s)) {
		martian.NewContext(req).SkipLogging()
	}
	return nil
}

func (s skipLoggingByHost) ModifyResponse(res *http.Response) error {
	return s.ModifyRequest(res.Request)
}
