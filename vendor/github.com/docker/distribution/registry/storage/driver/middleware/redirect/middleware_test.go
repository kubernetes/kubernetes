package middleware

import (
	"testing"

	check "gopkg.in/check.v1"
)

func Test(t *testing.T) { check.TestingT(t) }

type MiddlewareSuite struct{}

var _ = check.Suite(&MiddlewareSuite{})

func (s *MiddlewareSuite) TestNoConfig(c *check.C) {
	options := make(map[string]interface{})
	_, err := newRedirectStorageMiddleware(nil, options)
	c.Assert(err, check.ErrorMatches, "no baseurl provided")
}

func (s *MiddlewareSuite) TestMissingScheme(c *check.C) {
	options := make(map[string]interface{})
	options["baseurl"] = "example.com"
	_, err := newRedirectStorageMiddleware(nil, options)
	c.Assert(err, check.ErrorMatches, "no scheme specified for redirect baseurl")
}

func (s *MiddlewareSuite) TestHttpsPort(c *check.C) {
	options := make(map[string]interface{})
	options["baseurl"] = "https://example.com:5443"
	middleware, err := newRedirectStorageMiddleware(nil, options)
	c.Assert(err, check.Equals, nil)

	m, ok := middleware.(*redirectStorageMiddleware)
	c.Assert(ok, check.Equals, true)
	c.Assert(m.scheme, check.Equals, "https")
	c.Assert(m.host, check.Equals, "example.com:5443")

	url, err := middleware.URLFor(nil, "/rick/data", nil)
	c.Assert(err, check.Equals, nil)
	c.Assert(url, check.Equals, "https://example.com:5443/rick/data")
}

func (s *MiddlewareSuite) TestHTTP(c *check.C) {
	options := make(map[string]interface{})
	options["baseurl"] = "http://example.com"
	middleware, err := newRedirectStorageMiddleware(nil, options)
	c.Assert(err, check.Equals, nil)

	m, ok := middleware.(*redirectStorageMiddleware)
	c.Assert(ok, check.Equals, true)
	c.Assert(m.scheme, check.Equals, "http")
	c.Assert(m.host, check.Equals, "example.com")

	url, err := middleware.URLFor(nil, "morty/data", nil)
	c.Assert(err, check.Equals, nil)
	c.Assert(url, check.Equals, "http://example.com/morty/data")
}
