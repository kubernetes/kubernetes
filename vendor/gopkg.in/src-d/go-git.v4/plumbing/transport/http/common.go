// Package http implements the HTTP transport protocol.
package http

import (
	"bytes"
	"fmt"
	"net/http"
	"strconv"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/protocol/packp"
	"gopkg.in/src-d/go-git.v4/plumbing/transport"
)

// it requires a bytes.Buffer, because we need to know the length
func applyHeadersToRequest(req *http.Request, content *bytes.Buffer, host string, requestType string) {
	req.Header.Add("User-Agent", "git/1.0")
	req.Header.Add("Host", host) // host:port

	if content == nil {
		req.Header.Add("Accept", "*/*")
		return
	}

	req.Header.Add("Accept", fmt.Sprintf("application/x-%s-result", requestType))
	req.Header.Add("Content-Type", fmt.Sprintf("application/x-%s-request", requestType))
	req.Header.Add("Content-Length", strconv.Itoa(content.Len()))
}

func advertisedReferences(s *session, serviceName string) (*packp.AdvRefs, error) {
	url := fmt.Sprintf(
		"%s/info/refs?service=%s",
		s.endpoint.String(), serviceName,
	)

	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}

	s.applyAuthToRequest(req)
	applyHeadersToRequest(req, nil, s.endpoint.Host(), serviceName)
	res, err := s.client.Do(req)
	if err != nil {
		return nil, err
	}

	if err := NewErr(res); err != nil {
		_ = res.Body.Close()
		return nil, err
	}

	ar := packp.NewAdvRefs()
	if err := ar.Decode(res.Body); err != nil {
		if err == packp.ErrEmptyAdvRefs {
			err = transport.ErrEmptyRemoteRepository
		}

		return nil, err
	}

	transport.FilterUnsupportedCapabilities(ar.Capabilities)
	s.advRefs = ar

	return ar, nil
}

type client struct {
	c *http.Client
}

// DefaultClient is the default HTTP client, which uses `http.DefaultClient`.
var DefaultClient = NewClient(nil)

// NewClient creates a new client with a custom net/http client.
// See `InstallProtocol` to install and override default http client.
// Unless a properly initialized client is given, it will fall back into
// `http.DefaultClient`.
//
// Note that for HTTP client cannot distinguist between private repositories and
// unexistent repositories on GitHub. So it returns `ErrAuthorizationRequired`
// for both.
func NewClient(c *http.Client) transport.Transport {
	if c == nil {
		return &client{http.DefaultClient}
	}

	return &client{
		c: c,
	}
}

func (c *client) NewUploadPackSession(ep transport.Endpoint, auth transport.AuthMethod) (
	transport.UploadPackSession, error) {

	return newUploadPackSession(c.c, ep, auth)
}

func (c *client) NewReceivePackSession(ep transport.Endpoint, auth transport.AuthMethod) (
	transport.ReceivePackSession, error) {

	return newReceivePackSession(c.c, ep, auth)
}

type session struct {
	auth     AuthMethod
	client   *http.Client
	endpoint transport.Endpoint
	advRefs  *packp.AdvRefs
}

func newSession(c *http.Client, ep transport.Endpoint, auth transport.AuthMethod) (*session, error) {
	s := &session{
		auth:     basicAuthFromEndpoint(ep),
		client:   c,
		endpoint: ep,
	}
	if auth != nil {
		a, ok := auth.(AuthMethod)
		if !ok {
			return nil, transport.ErrInvalidAuthMethod
		}

		s.auth = a
	}

	return s, nil
}

func (*session) Close() error {
	return nil
}

func (s *session) applyAuthToRequest(req *http.Request) {
	if s.auth == nil {
		return
	}

	s.auth.setAuth(req)
}

// AuthMethod is concrete implementation of common.AuthMethod for HTTP services
type AuthMethod interface {
	transport.AuthMethod
	setAuth(r *http.Request)
}

func basicAuthFromEndpoint(ep transport.Endpoint) *BasicAuth {
	u := ep.User()
	if u == "" {
		return nil
	}

	return NewBasicAuth(u, ep.Password())
}

// BasicAuth represent a HTTP basic auth
type BasicAuth struct {
	username, password string
}

// NewBasicAuth returns a basicAuth base on the given user and password
func NewBasicAuth(username, password string) *BasicAuth {
	return &BasicAuth{username, password}
}

func (a *BasicAuth) setAuth(r *http.Request) {
	if a == nil {
		return
	}

	r.SetBasicAuth(a.username, a.password)
}

// Name is name of the auth
func (a *BasicAuth) Name() string {
	return "http-basic-auth"
}

func (a *BasicAuth) String() string {
	masked := "*******"
	if a.password == "" {
		masked = "<empty>"
	}

	return fmt.Sprintf("%s - %s:%s", a.Name(), a.username, masked)
}

// Err is a dedicated error to return errors based on status code
type Err struct {
	Response *http.Response
}

// NewErr returns a new Err based on a http response
func NewErr(r *http.Response) error {
	if r.StatusCode >= http.StatusOK && r.StatusCode < http.StatusMultipleChoices {
		return nil
	}

	switch r.StatusCode {
	case http.StatusUnauthorized:
		return transport.ErrAuthenticationRequired
	case http.StatusForbidden:
		return transport.ErrAuthorizationFailed
	case http.StatusNotFound:
		return transport.ErrRepositoryNotFound
	}

	return plumbing.NewUnexpectedError(&Err{r})
}

// StatusCode returns the status code of the response
func (e *Err) StatusCode() int {
	return e.Response.StatusCode
}

func (e *Err) Error() string {
	return fmt.Sprintf("unexpected requesting %q status code: %d",
		e.Response.Request.URL, e.Response.StatusCode,
	)
}
