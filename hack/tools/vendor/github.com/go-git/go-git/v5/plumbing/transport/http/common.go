// Package http implements the HTTP transport protocol.
package http

import (
	"bytes"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"strings"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/protocol/packp"
	"github.com/go-git/go-git/v5/plumbing/transport"
	"github.com/go-git/go-git/v5/utils/ioutil"
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

const infoRefsPath = "/info/refs"

func advertisedReferences(s *session, serviceName string) (ref *packp.AdvRefs, err error) {
	url := fmt.Sprintf(
		"%s%s?service=%s",
		s.endpoint.String(), infoRefsPath, serviceName,
	)

	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}

	s.ApplyAuthToRequest(req)
	applyHeadersToRequest(req, nil, s.endpoint.Host, serviceName)
	res, err := s.client.Do(req)
	if err != nil {
		return nil, err
	}

	s.ModifyEndpointIfRedirect(res)
	defer ioutil.CheckClose(res.Body, &err)

	if err = NewErr(res); err != nil {
		return nil, err
	}

	ar := packp.NewAdvRefs()
	if err = ar.Decode(res.Body); err != nil {
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
// Note that for HTTP client cannot distinguish between private repositories and
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

func (c *client) NewUploadPackSession(ep *transport.Endpoint, auth transport.AuthMethod) (
	transport.UploadPackSession, error) {

	return newUploadPackSession(c.c, ep, auth)
}

func (c *client) NewReceivePackSession(ep *transport.Endpoint, auth transport.AuthMethod) (
	transport.ReceivePackSession, error) {

	return newReceivePackSession(c.c, ep, auth)
}

type session struct {
	auth     AuthMethod
	client   *http.Client
	endpoint *transport.Endpoint
	advRefs  *packp.AdvRefs
}

func newSession(c *http.Client, ep *transport.Endpoint, auth transport.AuthMethod) (*session, error) {
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

func (s *session) ApplyAuthToRequest(req *http.Request) {
	if s.auth == nil {
		return
	}

	s.auth.SetAuth(req)
}

func (s *session) ModifyEndpointIfRedirect(res *http.Response) {
	if res.Request == nil {
		return
	}

	r := res.Request
	if !strings.HasSuffix(r.URL.Path, infoRefsPath) {
		return
	}

	h, p, err := net.SplitHostPort(r.URL.Host)
	if err != nil {
		h = r.URL.Host
	}
	if p != "" {
		port, err := strconv.Atoi(p)
		if err == nil {
			s.endpoint.Port = port
		}
	}
	s.endpoint.Host = h

	s.endpoint.Protocol = r.URL.Scheme
	s.endpoint.Path = r.URL.Path[:len(r.URL.Path)-len(infoRefsPath)]
}

func (*session) Close() error {
	return nil
}

// AuthMethod is concrete implementation of common.AuthMethod for HTTP services
type AuthMethod interface {
	transport.AuthMethod
	SetAuth(r *http.Request)
}

func basicAuthFromEndpoint(ep *transport.Endpoint) *BasicAuth {
	u := ep.User
	if u == "" {
		return nil
	}

	return &BasicAuth{u, ep.Password}
}

// BasicAuth represent a HTTP basic auth
type BasicAuth struct {
	Username, Password string
}

func (a *BasicAuth) SetAuth(r *http.Request) {
	if a == nil {
		return
	}

	r.SetBasicAuth(a.Username, a.Password)
}

// Name is name of the auth
func (a *BasicAuth) Name() string {
	return "http-basic-auth"
}

func (a *BasicAuth) String() string {
	masked := "*******"
	if a.Password == "" {
		masked = "<empty>"
	}

	return fmt.Sprintf("%s - %s:%s", a.Name(), a.Username, masked)
}

// TokenAuth implements an http.AuthMethod that can be used with http transport
// to authenticate with HTTP token authentication (also known as bearer
// authentication).
//
// IMPORTANT: If you are looking to use OAuth tokens with popular servers (e.g.
// GitHub, Bitbucket, GitLab) you should use BasicAuth instead. These servers
// use basic HTTP authentication, with the OAuth token as user or password.
// Check the documentation of your git server for details.
type TokenAuth struct {
	Token string
}

func (a *TokenAuth) SetAuth(r *http.Request) {
	if a == nil {
		return
	}
	r.Header.Add("Authorization", fmt.Sprintf("Bearer %s", a.Token))
}

// Name is name of the auth
func (a *TokenAuth) Name() string {
	return "http-token-auth"
}

func (a *TokenAuth) String() string {
	masked := "*******"
	if a.Token == "" {
		masked = "<empty>"
	}
	return fmt.Sprintf("%s - %s", a.Name(), masked)
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
