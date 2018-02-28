package autorest

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/Azure/go-autorest/autorest/adal"
)

const (
	bearerChallengeHeader       = "Www-Authenticate"
	bearer                      = "Bearer"
	tenantID                    = "tenantID"
	apiKeyAuthorizerHeader      = "Ocp-Apim-Subscription-Key"
	bingAPISdkHeader            = "X-BingApis-SDK-Client"
	golangBingAPISdkHeaderValue = "Go-SDK"
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

// APIKeyAuthorizer implements API Key authorization.
type APIKeyAuthorizer struct {
	headers         map[string]interface{}
	queryParameters map[string]interface{}
}

// NewAPIKeyAuthorizerWithHeaders creates an ApiKeyAuthorizer with headers.
func NewAPIKeyAuthorizerWithHeaders(headers map[string]interface{}) *APIKeyAuthorizer {
	return NewAPIKeyAuthorizer(headers, nil)
}

// NewAPIKeyAuthorizerWithQueryParameters creates an ApiKeyAuthorizer with query parameters.
func NewAPIKeyAuthorizerWithQueryParameters(queryParameters map[string]interface{}) *APIKeyAuthorizer {
	return NewAPIKeyAuthorizer(nil, queryParameters)
}

// NewAPIKeyAuthorizer creates an ApiKeyAuthorizer with headers.
func NewAPIKeyAuthorizer(headers map[string]interface{}, queryParameters map[string]interface{}) *APIKeyAuthorizer {
	return &APIKeyAuthorizer{headers: headers, queryParameters: queryParameters}
}

// WithAuthorization returns a PrepareDecorator that adds an HTTP headers and Query Paramaters
func (aka *APIKeyAuthorizer) WithAuthorization() PrepareDecorator {
	return func(p Preparer) Preparer {
		return DecoratePreparer(p, WithHeaders(aka.headers), WithQueryParameters(aka.queryParameters))
	}
}

// CognitiveServicesAuthorizer implements authorization for Cognitive Services.
type CognitiveServicesAuthorizer struct {
	subscriptionKey string
}

// NewCognitiveServicesAuthorizer is
func NewCognitiveServicesAuthorizer(subscriptionKey string) *CognitiveServicesAuthorizer {
	return &CognitiveServicesAuthorizer{subscriptionKey: subscriptionKey}
}

// WithAuthorization is
func (csa *CognitiveServicesAuthorizer) WithAuthorization() PrepareDecorator {
	headers := make(map[string]interface{})
	headers[apiKeyAuthorizerHeader] = csa.subscriptionKey
	headers[bingAPISdkHeader] = golangBingAPISdkHeaderValue

	return NewAPIKeyAuthorizerWithHeaders(headers).WithAuthorization()
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
					var resp *http.Response
					if tokError, ok := err.(adal.TokenRefreshError); ok {
						resp = tokError.Response()
					}
					return r, NewErrorWithError(err, "azure.BearerAuthorizer", "WithAuthorization", resp,
						"Failed to refresh the Token for request to %s", r.URL)
				}
			}
			return (ba.withBearerAuthorization()(p)).Prepare(r)
		})
	}
}

// BearerAuthorizerCallbackFunc is the authentication callback signature.
type BearerAuthorizerCallbackFunc func(tenantID, resource string) (*BearerAuthorizer, error)

// BearerAuthorizerCallback implements bearer authorization via a callback.
type BearerAuthorizerCallback struct {
	sender   Sender
	callback BearerAuthorizerCallbackFunc
}

// NewBearerAuthorizerCallback creates a bearer authorization callback.  The callback
// is invoked when the HTTP request is submitted.
func NewBearerAuthorizerCallback(sender Sender, callback BearerAuthorizerCallbackFunc) *BearerAuthorizerCallback {
	if sender == nil {
		sender = &http.Client{}
	}
	return &BearerAuthorizerCallback{sender: sender, callback: callback}
}

// WithAuthorization returns a PrepareDecorator that adds an HTTP Authorization header whose value
// is "Bearer " followed by the token.  The BearerAuthorizer is obtained via a user-supplied callback.
//
// By default, the token will be automatically refreshed through the Refresher interface.
func (bacb *BearerAuthorizerCallback) WithAuthorization() PrepareDecorator {
	return func(p Preparer) Preparer {
		return PreparerFunc(func(r *http.Request) (*http.Request, error) {
			// make a copy of the request and remove the body as it's not
			// required and avoids us having to create a copy of it.
			rCopy := *r
			removeRequestBody(&rCopy)

			resp, err := bacb.sender.Do(&rCopy)
			if err == nil && resp.StatusCode == 401 {
				defer resp.Body.Close()
				if hasBearerChallenge(resp) {
					bc, err := newBearerChallenge(resp)
					if err != nil {
						return r, err
					}
					if bacb.callback != nil {
						ba, err := bacb.callback(bc.values[tenantID], bc.values["resource"])
						if err != nil {
							return r, err
						}
						return ba.WithAuthorization()(p).Prepare(r)
					}
				}
			}
			return r, err
		})
	}
}

// returns true if the HTTP response contains a bearer challenge
func hasBearerChallenge(resp *http.Response) bool {
	authHeader := resp.Header.Get(bearerChallengeHeader)
	if len(authHeader) == 0 || strings.Index(authHeader, bearer) < 0 {
		return false
	}
	return true
}

type bearerChallenge struct {
	values map[string]string
}

func newBearerChallenge(resp *http.Response) (bc bearerChallenge, err error) {
	challenge := strings.TrimSpace(resp.Header.Get(bearerChallengeHeader))
	trimmedChallenge := challenge[len(bearer)+1:]

	// challenge is a set of key=value pairs that are comma delimited
	pairs := strings.Split(trimmedChallenge, ",")
	if len(pairs) < 1 {
		err = fmt.Errorf("challenge '%s' contains no pairs", challenge)
		return bc, err
	}

	bc.values = make(map[string]string)
	for i := range pairs {
		trimmedPair := strings.TrimSpace(pairs[i])
		pair := strings.Split(trimmedPair, "=")
		if len(pair) == 2 {
			// remove the enclosing quotes
			key := strings.Trim(pair[0], "\"")
			value := strings.Trim(pair[1], "\"")

			switch key {
			case "authorization", "authorization_uri":
				// strip the tenant ID from the authorization URL
				asURL, err := url.Parse(value)
				if err != nil {
					return bc, err
				}
				bc.values[tenantID] = asURL.Path[1:]
			default:
				bc.values[key] = value
			}
		}
	}

	return bc, err
}

// EventGridKeyAuthorizer implements authorization for event grid using key authentication.
type EventGridKeyAuthorizer struct {
	topicKey string
}

// NewEventGridKeyAuthorizer creates a new EventGridKeyAuthorizer
// with the specified topic key.
func NewEventGridKeyAuthorizer(topicKey string) EventGridKeyAuthorizer {
	return EventGridKeyAuthorizer{topicKey: topicKey}
}

// WithAuthorization returns a PrepareDecorator that adds the aeg-sas-key authentication header.
func (egta EventGridKeyAuthorizer) WithAuthorization() PrepareDecorator {
	headers := map[string]interface{}{
		"aeg-sas-key": egta.topicKey,
	}
	return NewAPIKeyAuthorizerWithHeaders(headers).WithAuthorization()
}
