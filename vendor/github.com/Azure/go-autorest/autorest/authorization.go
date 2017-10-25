package autorest

import (
	"fmt"
	"net/http"

	"github.com/Azure/go-autorest/autorest/adal"
)

// Authorizer is the interface that provides a PrepareDecorator used to supply request
// authorization. Most often, the Authorizer decorator runs last so it has access to the full
// state of the formed HTTP request.
type Authorizer interface {
	WithAuthorization() PrepareDecorator
}

// NullAuthorizer implements a default, "do nothing" Authorizer.
type NullAuthorizer struct{}

// WithAuthorization returns a PrepareDecorator that does nothing.
func (na NullAuthorizer) WithAuthorization() PrepareDecorator {
	return WithNothing()
}

// BearerAuthorizer implements the bearer authorization
type BearerAuthorizer struct {
	tokenProvider adal.OAuthTokenProvider
}

// NewBearerAuthorizer crates a BearerAuthorizer using the given token provider
func NewBearerAuthorizer(tp adal.OAuthTokenProvider) *BearerAuthorizer {
	return &BearerAuthorizer{tokenProvider: tp}
}

func (ba *BearerAuthorizer) withBearerAuthorization() PrepareDecorator {
	return WithHeader(headerAuthorization, fmt.Sprintf("Bearer %s", ba.tokenProvider.OAuthToken()))
}

// WithAuthorization returns a PrepareDecorator that adds an HTTP Authorization header whose
// value is "Bearer " followed by the token.
//
// By default, the token will be automatically refreshed through the Refresher interface.
func (ba *BearerAuthorizer) WithAuthorization() PrepareDecorator {
	return func(p Preparer) Preparer {
		return PreparerFunc(func(r *http.Request) (*http.Request, error) {
			refresher, ok := ba.tokenProvider.(adal.Refresher)
			if ok {
				err := refresher.EnsureFresh()
				if err != nil {
					return r, NewErrorWithError(err, "azure.BearerAuthorizer", "WithAuthorization", nil,
						"Failed to refresh the Token for request to %s", r.URL)
				}
			}
			return (ba.withBearerAuthorization()(p)).Prepare(r)
		})
	}
}
