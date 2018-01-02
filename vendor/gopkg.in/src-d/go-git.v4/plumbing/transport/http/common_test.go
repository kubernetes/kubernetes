package http

import (
	"crypto/tls"
	"net/http"
	"testing"

	"gopkg.in/src-d/go-git.v4/plumbing/transport"

	. "gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

type ClientSuite struct {
	Endpoint  transport.Endpoint
	EmptyAuth transport.AuthMethod
}

var _ = Suite(&ClientSuite{})

func (s *ClientSuite) SetUpSuite(c *C) {
	var err error
	s.Endpoint, err = transport.NewEndpoint(
		"https://github.com/git-fixtures/basic",
	)
	c.Assert(err, IsNil)
}

func (s *UploadPackSuite) TestNewClient(c *C) {
	roundTripper := &http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
	}
	cl := &http.Client{Transport: roundTripper}
	r, ok := NewClient(cl).(*client)
	c.Assert(ok, Equals, true)
	c.Assert(r.c, Equals, cl)
}

func (s *ClientSuite) TestNewBasicAuth(c *C) {
	a := NewBasicAuth("foo", "qux")

	c.Assert(a.Name(), Equals, "http-basic-auth")
	c.Assert(a.String(), Equals, "http-basic-auth - foo:*******")
}

func (s *ClientSuite) TestNewErrOK(c *C) {
	res := &http.Response{StatusCode: http.StatusOK}
	err := NewErr(res)
	c.Assert(err, IsNil)
}

func (s *ClientSuite) TestNewErrUnauthorized(c *C) {
	s.testNewHTTPError(c, http.StatusUnauthorized, "authentication required")
}

func (s *ClientSuite) TestNewErrForbidden(c *C) {
	s.testNewHTTPError(c, http.StatusForbidden, "authorization failed")
}

func (s *ClientSuite) TestNewErrNotFound(c *C) {
	s.testNewHTTPError(c, http.StatusNotFound, "repository not found")
}

func (s *ClientSuite) TestNewHTTPError40x(c *C) {
	s.testNewHTTPError(c, http.StatusPaymentRequired,
		"unexpected client error.*")
}

func (s *ClientSuite) testNewHTTPError(c *C, code int, msg string) {
	req, _ := http.NewRequest("GET", "foo", nil)
	res := &http.Response{
		StatusCode: code,
		Request:    req,
	}

	err := NewErr(res)
	c.Assert(err, NotNil)
	c.Assert(err, ErrorMatches, msg)
}

func (s *ClientSuite) TestSetAuth(c *C) {
	auth := &BasicAuth{}
	r, err := DefaultClient.NewUploadPackSession(s.Endpoint, auth)
	c.Assert(err, IsNil)
	c.Assert(auth, Equals, r.(*upSession).auth)
}

type mockAuth struct{}

func (*mockAuth) Name() string   { return "" }
func (*mockAuth) String() string { return "" }

func (s *ClientSuite) TestSetAuthWrongType(c *C) {
	_, err := DefaultClient.NewUploadPackSession(s.Endpoint, &mockAuth{})
	c.Assert(err, Equals, transport.ErrInvalidAuthMethod)
}
