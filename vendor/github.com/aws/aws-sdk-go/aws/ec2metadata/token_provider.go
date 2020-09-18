package ec2metadata

import (
	"net/http"
	"sync/atomic"
	"time"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/request"
)

// A tokenProvider struct provides access to EC2Metadata client
// and atomic instance of a token, along with configuredTTL for it.
// tokenProvider also provides an atomic flag to disable the
// fetch token operation.
// The disabled member will use 0 as false, and 1 as true.
type tokenProvider struct {
	client        *EC2Metadata
	token         atomic.Value
	configuredTTL time.Duration
	disabled      uint32
}

// A ec2Token struct helps use of token in EC2 Metadata service ops
type ec2Token struct {
	token string
	credentials.Expiry
}

// newTokenProvider provides a pointer to a tokenProvider instance
func newTokenProvider(c *EC2Metadata, duration time.Duration) *tokenProvider {
	return &tokenProvider{client: c, configuredTTL: duration}
}

// fetchTokenHandler fetches token for EC2Metadata service client by default.
func (t *tokenProvider) fetchTokenHandler(r *request.Request) {

	// short-circuits to insecure data flow if tokenProvider is disabled.
	if v := atomic.LoadUint32(&t.disabled); v == 1 {
		return
	}

	if ec2Token, ok := t.token.Load().(ec2Token); ok && !ec2Token.IsExpired() {
		r.HTTPRequest.Header.Set(tokenHeader, ec2Token.token)
		return
	}

	output, err := t.client.getToken(r.Context(), t.configuredTTL)

	if err != nil {

		// change the disabled flag on token provider to true,
		// when error is request timeout error.
		if requestFailureError, ok := err.(awserr.RequestFailure); ok {
			switch requestFailureError.StatusCode() {
			case http.StatusForbidden, http.StatusNotFound, http.StatusMethodNotAllowed:
				atomic.StoreUint32(&t.disabled, 1)
			case http.StatusBadRequest:
				r.Error = requestFailureError
			}

			// Check if request timed out while waiting for response
			if e, ok := requestFailureError.OrigErr().(awserr.Error); ok {
				if e.Code() == request.ErrCodeRequestError {
					atomic.StoreUint32(&t.disabled, 1)
				}
			}
		}
		return
	}

	newToken := ec2Token{
		token: output.Token,
	}
	newToken.SetExpiration(time.Now().Add(output.TTL), ttlExpirationWindow)
	t.token.Store(newToken)

	// Inject token header to the request.
	if ec2Token, ok := t.token.Load().(ec2Token); ok {
		r.HTTPRequest.Header.Set(tokenHeader, ec2Token.token)
	}
}

// enableTokenProviderHandler enables the token provider
func (t *tokenProvider) enableTokenProviderHandler(r *request.Request) {
	// If the error code status is 401, we enable the token provider
	if e, ok := r.Error.(awserr.RequestFailure); ok && e != nil &&
		e.StatusCode() == http.StatusUnauthorized {
		atomic.StoreUint32(&t.disabled, 0)
	}
}
