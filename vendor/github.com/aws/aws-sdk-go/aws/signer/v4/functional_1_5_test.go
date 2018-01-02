// +build go1.5

package v4_test

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws/signer/v4"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/stretchr/testify/assert"
)

func TestStandaloneSign(t *testing.T) {
	creds := unit.Session.Config.Credentials
	signer := v4.NewSigner(creds)

	for _, c := range standaloneSignCases {
		host := fmt.Sprintf("https://%s.%s.%s.amazonaws.com",
			c.SubDomain, c.Region, c.Service)

		req, err := http.NewRequest("GET", host, nil)
		assert.NoError(t, err)

		// URL.EscapedPath() will be used by the signer to get the
		// escaped form of the request's URI path.
		req.URL.Path = c.OrigURI
		req.URL.RawQuery = c.OrigQuery

		_, err = signer.Sign(req, nil, c.Service, c.Region, time.Unix(0, 0))
		assert.NoError(t, err)

		actual := req.Header.Get("Authorization")
		assert.Equal(t, c.ExpSig, actual)
		assert.Equal(t, c.OrigURI, req.URL.Path)
		assert.Equal(t, c.EscapedURI, req.URL.EscapedPath())
	}
}

func TestStandaloneSign_RawPath(t *testing.T) {
	creds := unit.Session.Config.Credentials
	signer := v4.NewSigner(creds)

	for _, c := range standaloneSignCases {
		host := fmt.Sprintf("https://%s.%s.%s.amazonaws.com",
			c.SubDomain, c.Region, c.Service)

		req, err := http.NewRequest("GET", host, nil)
		assert.NoError(t, err)

		// URL.EscapedPath() will be used by the signer to get the
		// escaped form of the request's URI path.
		req.URL.Path = c.OrigURI
		req.URL.RawPath = c.EscapedURI
		req.URL.RawQuery = c.OrigQuery

		_, err = signer.Sign(req, nil, c.Service, c.Region, time.Unix(0, 0))
		assert.NoError(t, err)

		actual := req.Header.Get("Authorization")
		assert.Equal(t, c.ExpSig, actual)
		assert.Equal(t, c.OrigURI, req.URL.Path)
		assert.Equal(t, c.EscapedURI, req.URL.EscapedPath())
	}
}
