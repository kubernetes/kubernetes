// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package azidext

import (
	"errors"
	"net/http"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/runtime"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"github.com/Azure/go-autorest/autorest"
)

// NewTokenCredentialAdapter is used to adapt an azcore.TokenCredential to an autorest.Authorizer
func NewTokenCredentialAdapter(credential azcore.TokenCredential, scopes []string) autorest.Authorizer {
	tkPolicy := runtime.NewBearerTokenPolicy(credential, scopes, nil)
	return &policyAdapter{
		pl: runtime.NewPipeline("azidext", "v0.4.0", runtime.PipelineOptions{
			PerRetry: []policy.Policy{tkPolicy, nullPolicy{}},
		}, nil),
	}
}

type policyAdapter struct {
	pl runtime.Pipeline
}

// WithAuthorization implements the autorest.Authorizer interface for type policyAdapter.
func (ca *policyAdapter) WithAuthorization() autorest.PrepareDecorator {
	return func(p autorest.Preparer) autorest.Preparer {
		return autorest.PreparerFunc(func(r *http.Request) (*http.Request, error) {
			r, err := p.Prepare(r)
			if err != nil {
				return r, err
			}
			// create a dummy request
			req, err := runtime.NewRequest(r.Context(), r.Method, r.URL.String())
			if err != nil {
				return r, err
			}
			_, err = ca.pl.Do(req)
			// if the authentication failed due to invalid/missing credentials
			// return a wrapped error so the retry policy won't kick in.
			type nonRetriable interface {
				NonRetriable()
			}
			var nre nonRetriable
			if errors.As(err, &nre) {
				return r, &tokenRefreshError{
					inner: err,
				}
			}
			// some other error
			if err != nil {
				return r, err
			}
			// copy the authorization header to the real request
			const authHeader = "Authorization"
			r.Header.Set(authHeader, req.Raw().Header.Get(authHeader))
			return r, err
		})
	}
}

// DefaultManagementScope is the default credential scope for Azure Resource Management.
const DefaultManagementScope = "https://management.azure.com//.default"

// DefaultAzureCredentialOptions contains credential and authentication policy options.
type DefaultAzureCredentialOptions struct {
	// DefaultCredential contains configuration options passed to azidentity.NewDefaultAzureCredential().
	// Set this to nil to accept the underlying default behavior.
	DefaultCredential *azidentity.DefaultAzureCredentialOptions

	// Scopes contains the list of permission scopes required for the token.
	// Setting this to nil will use the DefaultManagementScope when acquiring a token.
	Scopes []string
}

// NewDefaultAzureCredentialAdapter adapts azcore.NewDefaultAzureCredential to an autorest.Authorizer.
func NewDefaultAzureCredentialAdapter(options *DefaultAzureCredentialOptions) (autorest.Authorizer, error) {
	if options == nil {
		options = &DefaultAzureCredentialOptions{
			Scopes: []string{DefaultManagementScope},
		}
	}
	cred, err := azidentity.NewDefaultAzureCredential(options.DefaultCredential)
	if err != nil {
		return nil, err
	}
	return NewTokenCredentialAdapter(cred, options.Scopes), nil
}

// dummy policy to terminate the pipeline
type nullPolicy struct{}

func (nullPolicy) Do(req *policy.Request) (*http.Response, error) {
	return &http.Response{StatusCode: http.StatusOK}, nil
}

// error type returned to prevent the retry policy from retrying the request
type tokenRefreshError struct {
	inner error
}

func (t *tokenRefreshError) Error() string {
	return t.inner.Error()
}

func (t *tokenRefreshError) Response() *http.Response {
	return nil
}

func (t *tokenRefreshError) Unwrap() error {
	return t.inner
}
