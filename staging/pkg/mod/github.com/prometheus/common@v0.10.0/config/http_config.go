// Copyright 2016 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build go1.8

package config

import (
	"bytes"
	"crypto/md5"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/mwitkow/go-conntrack"
	"golang.org/x/net/http2"
	"gopkg.in/yaml.v2"
)

type closeIdler interface {
	CloseIdleConnections()
}

// BasicAuth contains basic HTTP authentication credentials.
type BasicAuth struct {
	Username     string `yaml:"username"`
	Password     Secret `yaml:"password,omitempty"`
	PasswordFile string `yaml:"password_file,omitempty"`
}

// URL is a custom URL type that allows validation at configuration load time.
type URL struct {
	*url.URL
}

// UnmarshalYAML implements the yaml.Unmarshaler interface for URLs.
func (u *URL) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var s string
	if err := unmarshal(&s); err != nil {
		return err
	}

	urlp, err := url.Parse(s)
	if err != nil {
		return err
	}
	u.URL = urlp
	return nil
}

// MarshalYAML implements the yaml.Marshaler interface for URLs.
func (u URL) MarshalYAML() (interface{}, error) {
	if u.URL != nil {
		return u.String(), nil
	}
	return nil, nil
}

// HTTPClientConfig configures an HTTP client.
type HTTPClientConfig struct {
	// The HTTP basic authentication credentials for the targets.
	BasicAuth *BasicAuth `yaml:"basic_auth,omitempty"`
	// The bearer token for the targets.
	BearerToken Secret `yaml:"bearer_token,omitempty"`
	// The bearer token file for the targets.
	BearerTokenFile string `yaml:"bearer_token_file,omitempty"`
	// HTTP proxy server to use to connect to the targets.
	ProxyURL URL `yaml:"proxy_url,omitempty"`
	// TLSConfig to use to connect to the targets.
	TLSConfig TLSConfig `yaml:"tls_config,omitempty"`
}

// Validate validates the HTTPClientConfig to check only one of BearerToken,
// BasicAuth and BearerTokenFile is configured.
func (c *HTTPClientConfig) Validate() error {
	if len(c.BearerToken) > 0 && len(c.BearerTokenFile) > 0 {
		return fmt.Errorf("at most one of bearer_token & bearer_token_file must be configured")
	}
	if c.BasicAuth != nil && (len(c.BearerToken) > 0 || len(c.BearerTokenFile) > 0) {
		return fmt.Errorf("at most one of basic_auth, bearer_token & bearer_token_file must be configured")
	}
	if c.BasicAuth != nil && (string(c.BasicAuth.Password) != "" && c.BasicAuth.PasswordFile != "") {
		return fmt.Errorf("at most one of basic_auth password & password_file must be configured")
	}
	return nil
}

// UnmarshalYAML implements the yaml.Unmarshaler interface
func (c *HTTPClientConfig) UnmarshalYAML(unmarshal func(interface{}) error) error {
	type plain HTTPClientConfig
	if err := unmarshal((*plain)(c)); err != nil {
		return err
	}
	return c.Validate()
}

// UnmarshalYAML implements the yaml.Unmarshaler interface.
func (a *BasicAuth) UnmarshalYAML(unmarshal func(interface{}) error) error {
	type plain BasicAuth
	return unmarshal((*plain)(a))
}

// NewClient returns a http.Client using the specified http.RoundTripper.
func newClient(rt http.RoundTripper) *http.Client {
	return &http.Client{Transport: rt}
}

// NewClientFromConfig returns a new HTTP client configured for the
// given config.HTTPClientConfig. The name is used as go-conntrack metric label.
func NewClientFromConfig(cfg HTTPClientConfig, name string, disableKeepAlives bool) (*http.Client, error) {
	rt, err := NewRoundTripperFromConfig(cfg, name, disableKeepAlives)
	if err != nil {
		return nil, err
	}
	return newClient(rt), nil
}

// NewRoundTripperFromConfig returns a new HTTP RoundTripper configured for the
// given config.HTTPClientConfig. The name is used as go-conntrack metric label.
func NewRoundTripperFromConfig(cfg HTTPClientConfig, name string, disableKeepAlives bool) (http.RoundTripper, error) {
	newRT := func(tlsConfig *tls.Config) (http.RoundTripper, error) {
		// The only timeout we care about is the configured scrape timeout.
		// It is applied on request. So we leave out any timings here.
		var rt http.RoundTripper = &http.Transport{
			Proxy:               http.ProxyURL(cfg.ProxyURL.URL),
			MaxIdleConns:        20000,
			MaxIdleConnsPerHost: 1000, // see https://github.com/golang/go/issues/13801
			DisableKeepAlives:   disableKeepAlives,
			TLSClientConfig:     tlsConfig,
			DisableCompression:  true,
			// 5 minutes is typically above the maximum sane scrape interval. So we can
			// use keepalive for all configurations.
			IdleConnTimeout:       5 * time.Minute,
			TLSHandshakeTimeout:   10 * time.Second,
			ExpectContinueTimeout: 1 * time.Second,
			DialContext: conntrack.NewDialContextFunc(
				conntrack.DialWithTracing(),
				conntrack.DialWithName(name),
			),
		}
		// TODO: use ForceAttemptHTTP2 when we move to Go 1.13+.
		err := http2.ConfigureTransport(rt.(*http.Transport))
		if err != nil {
			return nil, err
		}

		// If a bearer token is provided, create a round tripper that will set the
		// Authorization header correctly on each request.
		if len(cfg.BearerToken) > 0 {
			rt = NewBearerAuthRoundTripper(cfg.BearerToken, rt)
		} else if len(cfg.BearerTokenFile) > 0 {
			rt = NewBearerAuthFileRoundTripper(cfg.BearerTokenFile, rt)
		}

		if cfg.BasicAuth != nil {
			rt = NewBasicAuthRoundTripper(cfg.BasicAuth.Username, cfg.BasicAuth.Password, cfg.BasicAuth.PasswordFile, rt)
		}
		// Return a new configured RoundTripper.
		return rt, nil
	}

	tlsConfig, err := NewTLSConfig(&cfg.TLSConfig)
	if err != nil {
		return nil, err
	}

	if len(cfg.TLSConfig.CAFile) == 0 {
		// No need for a RoundTripper that reloads the CA file automatically.
		return newRT(tlsConfig)
	}

	return newTLSRoundTripper(tlsConfig, cfg.TLSConfig.CAFile, newRT)
}

type bearerAuthRoundTripper struct {
	bearerToken Secret
	rt          http.RoundTripper
}

// NewBearerAuthRoundTripper adds the provided bearer token to a request unless the authorization
// header has already been set.
func NewBearerAuthRoundTripper(token Secret, rt http.RoundTripper) http.RoundTripper {
	return &bearerAuthRoundTripper{token, rt}
}

func (rt *bearerAuthRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if len(req.Header.Get("Authorization")) == 0 {
		req = cloneRequest(req)
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", string(rt.bearerToken)))
	}
	return rt.rt.RoundTrip(req)
}

func (rt *bearerAuthRoundTripper) CloseIdleConnections() {
	if ci, ok := rt.rt.(closeIdler); ok {
		ci.CloseIdleConnections()
	}
}

type bearerAuthFileRoundTripper struct {
	bearerFile string
	rt         http.RoundTripper
}

// NewBearerAuthFileRoundTripper adds the bearer token read from the provided file to a request unless
// the authorization header has already been set. This file is read for every request.
func NewBearerAuthFileRoundTripper(bearerFile string, rt http.RoundTripper) http.RoundTripper {
	return &bearerAuthFileRoundTripper{bearerFile, rt}
}

func (rt *bearerAuthFileRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if len(req.Header.Get("Authorization")) == 0 {
		b, err := ioutil.ReadFile(rt.bearerFile)
		if err != nil {
			return nil, fmt.Errorf("unable to read bearer token file %s: %s", rt.bearerFile, err)
		}
		bearerToken := strings.TrimSpace(string(b))

		req = cloneRequest(req)
		req.Header.Set("Authorization", "Bearer "+bearerToken)
	}

	return rt.rt.RoundTrip(req)
}

func (rt *bearerAuthFileRoundTripper) CloseIdleConnections() {
	if ci, ok := rt.rt.(closeIdler); ok {
		ci.CloseIdleConnections()
	}
}

type basicAuthRoundTripper struct {
	username     string
	password     Secret
	passwordFile string
	rt           http.RoundTripper
}

// NewBasicAuthRoundTripper will apply a BASIC auth authorization header to a request unless it has
// already been set.
func NewBasicAuthRoundTripper(username string, password Secret, passwordFile string, rt http.RoundTripper) http.RoundTripper {
	return &basicAuthRoundTripper{username, password, passwordFile, rt}
}

func (rt *basicAuthRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if len(req.Header.Get("Authorization")) != 0 {
		return rt.rt.RoundTrip(req)
	}
	req = cloneRequest(req)
	if rt.passwordFile != "" {
		bs, err := ioutil.ReadFile(rt.passwordFile)
		if err != nil {
			return nil, fmt.Errorf("unable to read basic auth password file %s: %s", rt.passwordFile, err)
		}
		req.SetBasicAuth(rt.username, strings.TrimSpace(string(bs)))
	} else {
		req.SetBasicAuth(rt.username, strings.TrimSpace(string(rt.password)))
	}
	return rt.rt.RoundTrip(req)
}

func (rt *basicAuthRoundTripper) CloseIdleConnections() {
	if ci, ok := rt.rt.(closeIdler); ok {
		ci.CloseIdleConnections()
	}
}

// cloneRequest returns a clone of the provided *http.Request.
// The clone is a shallow copy of the struct and its Header map.
func cloneRequest(r *http.Request) *http.Request {
	// Shallow copy of the struct.
	r2 := new(http.Request)
	*r2 = *r
	// Deep copy of the Header.
	r2.Header = make(http.Header)
	for k, s := range r.Header {
		r2.Header[k] = s
	}
	return r2
}

// NewTLSConfig creates a new tls.Config from the given TLSConfig.
func NewTLSConfig(cfg *TLSConfig) (*tls.Config, error) {
	tlsConfig := &tls.Config{InsecureSkipVerify: cfg.InsecureSkipVerify}

	// If a CA cert is provided then let's read it in so we can validate the
	// scrape target's certificate properly.
	if len(cfg.CAFile) > 0 {
		b, err := readCAFile(cfg.CAFile)
		if err != nil {
			return nil, err
		}
		if !updateRootCA(tlsConfig, b) {
			return nil, fmt.Errorf("unable to use specified CA cert %s", cfg.CAFile)
		}
	}

	if len(cfg.ServerName) > 0 {
		tlsConfig.ServerName = cfg.ServerName
	}
	// If a client cert & key is provided then configure TLS config accordingly.
	if len(cfg.CertFile) > 0 && len(cfg.KeyFile) == 0 {
		return nil, fmt.Errorf("client cert file %q specified without client key file", cfg.CertFile)
	} else if len(cfg.KeyFile) > 0 && len(cfg.CertFile) == 0 {
		return nil, fmt.Errorf("client key file %q specified without client cert file", cfg.KeyFile)
	} else if len(cfg.CertFile) > 0 && len(cfg.KeyFile) > 0 {
		// Verify that client cert and key are valid.
		if _, err := cfg.getClientCertificate(nil); err != nil {
			return nil, err
		}
		tlsConfig.GetClientCertificate = cfg.getClientCertificate
	}

	return tlsConfig, nil
}

// TLSConfig configures the options for TLS connections.
type TLSConfig struct {
	// The CA cert to use for the targets.
	CAFile string `yaml:"ca_file,omitempty"`
	// The client cert file for the targets.
	CertFile string `yaml:"cert_file,omitempty"`
	// The client key file for the targets.
	KeyFile string `yaml:"key_file,omitempty"`
	// Used to verify the hostname for the targets.
	ServerName string `yaml:"server_name,omitempty"`
	// Disable target certificate validation.
	InsecureSkipVerify bool `yaml:"insecure_skip_verify"`
}

// UnmarshalYAML implements the yaml.Unmarshaler interface.
func (c *TLSConfig) UnmarshalYAML(unmarshal func(interface{}) error) error {
	type plain TLSConfig
	return unmarshal((*plain)(c))
}

// getClientCertificate reads the pair of client cert and key from disk and returns a tls.Certificate.
func (c *TLSConfig) getClientCertificate(*tls.CertificateRequestInfo) (*tls.Certificate, error) {
	cert, err := tls.LoadX509KeyPair(c.CertFile, c.KeyFile)
	if err != nil {
		return nil, fmt.Errorf("unable to use specified client cert (%s) & key (%s): %s", c.CertFile, c.KeyFile, err)
	}
	return &cert, nil
}

// readCAFile reads the CA cert file from disk.
func readCAFile(f string) ([]byte, error) {
	data, err := ioutil.ReadFile(f)
	if err != nil {
		return nil, fmt.Errorf("unable to load specified CA cert %s: %s", f, err)
	}
	return data, nil
}

// updateRootCA parses the given byte slice as a series of PEM encoded certificates and updates tls.Config.RootCAs.
func updateRootCA(cfg *tls.Config, b []byte) bool {
	caCertPool := x509.NewCertPool()
	if !caCertPool.AppendCertsFromPEM(b) {
		return false
	}
	cfg.RootCAs = caCertPool
	return true
}

// tlsRoundTripper is a RoundTripper that updates automatically its TLS
// configuration whenever the content of the CA file changes.
type tlsRoundTripper struct {
	caFile string
	// newRT returns a new RoundTripper.
	newRT func(*tls.Config) (http.RoundTripper, error)

	mtx        sync.RWMutex
	rt         http.RoundTripper
	hashCAFile []byte
	tlsConfig  *tls.Config
}

func newTLSRoundTripper(
	cfg *tls.Config,
	caFile string,
	newRT func(*tls.Config) (http.RoundTripper, error),
) (http.RoundTripper, error) {
	t := &tlsRoundTripper{
		caFile:    caFile,
		newRT:     newRT,
		tlsConfig: cfg,
	}

	rt, err := t.newRT(t.tlsConfig)
	if err != nil {
		return nil, err
	}
	t.rt = rt

	_, t.hashCAFile, err = t.getCAWithHash()
	if err != nil {
		return nil, err
	}

	return t, nil
}

func (t *tlsRoundTripper) getCAWithHash() ([]byte, []byte, error) {
	b, err := readCAFile(t.caFile)
	if err != nil {
		return nil, nil, err
	}
	h := md5.Sum(b)
	return b, h[:], nil

}

// RoundTrip implements the http.RoundTrip interface.
func (t *tlsRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	b, h, err := t.getCAWithHash()
	if err != nil {
		return nil, err
	}

	t.mtx.RLock()
	equal := bytes.Equal(h[:], t.hashCAFile)
	rt := t.rt
	t.mtx.RUnlock()
	if equal {
		// The CA cert hasn't changed, use the existing RoundTripper.
		return rt.RoundTrip(req)
	}

	// Create a new RoundTripper.
	tlsConfig := t.tlsConfig.Clone()
	if !updateRootCA(tlsConfig, b) {
		return nil, fmt.Errorf("unable to use specified CA cert %s", t.caFile)
	}
	rt, err = t.newRT(tlsConfig)
	if err != nil {
		return nil, err
	}
	t.CloseIdleConnections()

	t.mtx.Lock()
	t.rt = rt
	t.hashCAFile = h[:]
	t.mtx.Unlock()

	return rt.RoundTrip(req)
}

func (t *tlsRoundTripper) CloseIdleConnections() {
	t.mtx.RLock()
	defer t.mtx.RUnlock()
	if ci, ok := t.rt.(closeIdler); ok {
		ci.CloseIdleConnections()
	}
}

func (c HTTPClientConfig) String() string {
	b, err := yaml.Marshal(c)
	if err != nil {
		return fmt.Sprintf("<error creating http client config string: %s>", err)
	}
	return string(b)
}
