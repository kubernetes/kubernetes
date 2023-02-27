// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package runtime

import (
	"net/http"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/shared"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/internal/temporal"
)

// BearerTokenPolicy authorizes requests with bearer tokens acquired from a TokenCredential.
type BearerTokenPolicy struct {
	// mainResource is the resource to be retreived using the tenant specified in the credential
	mainResource *temporal.Resource[azcore.AccessToken, acquiringResourceState]
	// the following fields are read-only
	cred   azcore.TokenCredential
	scopes []string
}

type acquiringResourceState struct {
	req *policy.Request
	p   *BearerTokenPolicy
}

// acquire acquires or updates the resource; only one
// thread/goroutine at a time ever calls this function
func acquire(state acquiringResourceState) (newResource azcore.AccessToken, newExpiration time.Time, err error) {
	tk, err := state.p.cred.GetToken(state.req.Raw().Context(), policy.TokenRequestOptions{Scopes: state.p.scopes})
	if err != nil {
		return azcore.AccessToken{}, time.Time{}, err
	}
	return tk, tk.ExpiresOn, nil
}

// NewBearerTokenPolicy creates a policy object that authorizes requests with bearer tokens.
// cred: an azcore.TokenCredential implementation such as a credential object from azidentity
// scopes: the list of permission scopes required for the token.
// opts: optional settings. Pass nil to accept default values; this is the same as passing a zero-value options.
func NewBearerTokenPolicy(cred azcore.TokenCredential, scopes []string, opts *policy.BearerTokenOptions) *BearerTokenPolicy {
	return &BearerTokenPolicy{
		cred:         cred,
		scopes:       scopes,
		mainResource: temporal.NewResource(acquire),
	}
}

// Do authorizes a request with a bearer token
func (b *BearerTokenPolicy) Do(req *policy.Request) (*http.Response, error) {
	as := acquiringResourceState{
		p:   b,
		req: req,
	}
	tk, err := b.mainResource.Get(as)
	if err != nil {
		return nil, err
	}
	req.Raw().Header.Set(shared.HeaderAuthorization, shared.BearerTokenPrefix+tk.Token)
	return req.Next()
}
