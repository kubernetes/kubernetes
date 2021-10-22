// +build go1.5

package v4_test

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws/signer/v4"
	"github.com/aws/aws-sdk-go/awstesting/unit"
)

func TestStandaloneSign(t *testing.T) {
	creds := unit.Session.Config.Credentials
	signer := v4.NewSigner(creds)

	for _, c := range standaloneSignCases {
		host := fmt.Sprintf("https://%s.%s.%s.amazonaws.com",
			c.SubDomain, c.Region, c.Service)

		req, err := http.NewRequest("GET", host, nil)
		if err != nil {
			t.Errorf("expected no error, but received %v", err)
		}

		// URL.EscapedPath() will be used by the signer to get the
		// escaped form of the request's URI path.
		req.URL.Path = c.OrigURI
		req.URL.RawQuery = c.OrigQuery

		_, err = signer.Sign(req, nil, c.Service, c.Region, time.Unix(0, 0))
		if err != nil {
			t.Errorf("expected no error, but received %v", err)
		}

		actual := req.Header.Get("Authorization")
		if e, a := c.ExpSig, actual; e != a {
			t.Errorf("expected %v, but received %v", e, a)
		}
		if e, a := c.OrigURI, req.URL.Path; e != a {
			t.Errorf("expected %v, but received %v", e, a)
		}
		if e, a := c.EscapedURI, req.URL.EscapedPath(); e != a {
			t.Errorf("expected %v, but received %v", e, a)
		}
	}
}

func TestStandaloneSign_RawPath(t *testing.T) {
	creds := unit.Session.Config.Credentials
	signer := v4.NewSigner(creds)

	for _, c := range standaloneSignCases {
		host := fmt.Sprintf("https://%s.%s.%s.amazonaws.com",
			c.SubDomain, c.Region, c.Service)

		req, err := http.NewRequest("GET", host, nil)
		if err != nil {
			t.Errorf("expected no error, but received %v", err)
		}

		// URL.EscapedPath() will be used by the signer to get the
		// escaped form of the request's URI path.
		req.URL.Path = c.OrigURI
		req.URL.RawPath = c.EscapedURI
		req.URL.RawQuery = c.OrigQuery

		_, err = signer.Sign(req, nil, c.Service, c.Region, time.Unix(0, 0))
		if err != nil {
			t.Errorf("expected no error, but received %v", err)
		}

		actual := req.Header.Get("Authorization")
		if e, a := c.ExpSig, actual; e != a {
			t.Errorf("expected %v, but received %v", e, a)
		}
		if e, a := c.OrigURI, req.URL.Path; e != a {
			t.Errorf("expected %v, but received %v", e, a)
		}
		if e, a := c.EscapedURI, req.URL.EscapedPath(); e != a {
			t.Errorf("expected %v, but received %v", e, a)
		}
	}
}
