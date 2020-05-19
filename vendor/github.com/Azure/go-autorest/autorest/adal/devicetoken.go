package adal

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

/*
  This file is largely based on rjw57/oauth2device's code, with the follow differences:
   * scope -> resource, and only allow a single one
   * receive "Message" in the DeviceCode struct and show it to users as the prompt
   * azure-xplat-cli has the following behavior that this emulates:
     - does not send client_secret during the token exchange
     - sends resource again in the token exchange request
*/

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const (
	logPrefix = "autorest/adal/devicetoken:"
)

var (
	// ErrDeviceGeneric represents an unknown error from the token endpoint when using device flow
	ErrDeviceGeneric = fmt.Errorf("%s Error while retrieving OAuth token: Unknown Error", logPrefix)

	// ErrDeviceAccessDenied represents an access denied error from the token endpoint when using device flow
	ErrDeviceAccessDenied = fmt.Errorf("%s Error while retrieving OAuth token: Access Denied", logPrefix)

	// ErrDeviceAuthorizationPending represents the server waiting on the user to complete the device flow
	ErrDeviceAuthorizationPending = fmt.Errorf("%s Error while retrieving OAuth token: Authorization Pending", logPrefix)

	// ErrDeviceCodeExpired represents the server timing out and expiring the code during device flow
	ErrDeviceCodeExpired = fmt.Errorf("%s Error while retrieving OAuth token: Code Expired", logPrefix)

	// ErrDeviceSlowDown represents the service telling us we're polling too often during device flow
	ErrDeviceSlowDown = fmt.Errorf("%s Error while retrieving OAuth token: Slow Down", logPrefix)

	// ErrDeviceCodeEmpty represents an empty device code from the device endpoint while using device flow
	ErrDeviceCodeEmpty = fmt.Errorf("%s Error while retrieving device code: Device Code Empty", logPrefix)

	// ErrOAuthTokenEmpty represents an empty OAuth token from the token endpoint when using device flow
	ErrOAuthTokenEmpty = fmt.Errorf("%s Error while retrieving OAuth token: Token Empty", logPrefix)

	errCodeSendingFails   = "Error occurred while sending request for Device Authorization Code"
	errCodeHandlingFails  = "Error occurred while handling response from the Device Endpoint"
	errTokenSendingFails  = "Error occurred while sending request with device code for a token"
	errTokenHandlingFails = "Error occurred while handling response from the Token Endpoint (during device flow)"
	errStatusNotOK        = "Error HTTP status != 200"
)

// DeviceCode is the object returned by the device auth endpoint
// It contains information to instruct the user to complete the auth flow
type DeviceCode struct {
	DeviceCode      *string `json:"device_code,omitempty"`
	UserCode        *string `json:"user_code,omitempty"`
	VerificationURL *string `json:"verification_url,omitempty"`
	ExpiresIn       *int64  `json:"expires_in,string,omitempty"`
	Interval        *int64  `json:"interval,string,omitempty"`

	Message     *string `json:"message"` // Azure specific
	Resource    string  // store the following, stored when initiating, used when exchanging
	OAuthConfig OAuthConfig
	ClientID    string
}

// TokenError is the object returned by the token exchange endpoint
// when something is amiss
type TokenError struct {
	Error            *string `json:"error,omitempty"`
	ErrorCodes       []int   `json:"error_codes,omitempty"`
	ErrorDescription *string `json:"error_description,omitempty"`
	Timestamp        *string `json:"timestamp,omitempty"`
	TraceID          *string `json:"trace_id,omitempty"`
}

// DeviceToken is the object return by the token exchange endpoint
// It can either look like a Token or an ErrorToken, so put both here
// and check for presence of "Error" to know if we are in error state
type deviceToken struct {
	Token
	TokenError
}

// InitiateDeviceAuth initiates a device auth flow. It returns a DeviceCode
// that can be used with CheckForUserCompletion or WaitForUserCompletion.
// Deprecated: use InitiateDeviceAuthWithContext() instead.
func InitiateDeviceAuth(sender Sender, oauthConfig OAuthConfig, clientID, resource string) (*DeviceCode, error) {
	return InitiateDeviceAuthWithContext(context.Background(), sender, oauthConfig, clientID, resource)
}

// InitiateDeviceAuthWithContext initiates a device auth flow. It returns a DeviceCode
// that can be used with CheckForUserCompletion or WaitForUserCompletion.
func InitiateDeviceAuthWithContext(ctx context.Context, sender Sender, oauthConfig OAuthConfig, clientID, resource string) (*DeviceCode, error) {
	v := url.Values{
		"client_id": []string{clientID},
		"resource":  []string{resource},
	}

	s := v.Encode()
	body := ioutil.NopCloser(strings.NewReader(s))

	req, err := http.NewRequest(http.MethodPost, oauthConfig.DeviceCodeEndpoint.String(), body)
	if err != nil {
		return nil, fmt.Errorf("%s %s: %s", logPrefix, errCodeSendingFails, err.Error())
	}

	req.ContentLength = int64(len(s))
	req.Header.Set(contentType, mimeTypeFormPost)
	resp, err := sender.Do(req.WithContext(ctx))
	if err != nil {
		return nil, fmt.Errorf("%s %s: %s", logPrefix, errCodeSendingFails, err.Error())
	}
	defer resp.Body.Close()

	rb, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("%s %s: %s", logPrefix, errCodeHandlingFails, err.Error())
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s %s: %s", logPrefix, errCodeHandlingFails, errStatusNotOK)
	}

	if len(strings.Trim(string(rb), " ")) == 0 {
		return nil, ErrDeviceCodeEmpty
	}

	var code DeviceCode
	err = json.Unmarshal(rb, &code)
	if err != nil {
		return nil, fmt.Errorf("%s %s: %s", logPrefix, errCodeHandlingFails, err.Error())
	}

	code.ClientID = clientID
	code.Resource = resource
	code.OAuthConfig = oauthConfig

	return &code, nil
}

// CheckForUserCompletion takes a DeviceCode and checks with the Azure AD OAuth endpoint
// to see if the device flow has: been completed, timed out, or otherwise failed
// Deprecated: use CheckForUserCompletionWithContext() instead.
func CheckForUserCompletion(sender Sender, code *DeviceCode) (*Token, error) {
	return CheckForUserCompletionWithContext(context.Background(), sender, code)
}

// CheckForUserCompletionWithContext takes a DeviceCode and checks with the Azure AD OAuth endpoint
// to see if the device flow has: been completed, timed out, or otherwise failed
func CheckForUserCompletionWithContext(ctx context.Context, sender Sender, code *DeviceCode) (*Token, error) {
	v := url.Values{
		"client_id":  []string{code.ClientID},
		"code":       []string{*code.DeviceCode},
		"grant_type": []string{OAuthGrantTypeDeviceCode},
		"resource":   []string{code.Resource},
	}

	s := v.Encode()
	body := ioutil.NopCloser(strings.NewReader(s))

	req, err := http.NewRequest(http.MethodPost, code.OAuthConfig.TokenEndpoint.String(), body)
	if err != nil {
		return nil, fmt.Errorf("%s %s: %s", logPrefix, errTokenSendingFails, err.Error())
	}

	req.ContentLength = int64(len(s))
	req.Header.Set(contentType, mimeTypeFormPost)
	resp, err := sender.Do(req.WithContext(ctx))
	if err != nil {
		return nil, fmt.Errorf("%s %s: %s", logPrefix, errTokenSendingFails, err.Error())
	}
	defer resp.Body.Close()

	rb, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("%s %s: %s", logPrefix, errTokenHandlingFails, err.Error())
	}

	if resp.StatusCode != http.StatusOK && len(strings.Trim(string(rb), " ")) == 0 {
		return nil, fmt.Errorf("%s %s: %s", logPrefix, errTokenHandlingFails, errStatusNotOK)
	}
	if len(strings.Trim(string(rb), " ")) == 0 {
		return nil, ErrOAuthTokenEmpty
	}

	var token deviceToken
	err = json.Unmarshal(rb, &token)
	if err != nil {
		return nil, fmt.Errorf("%s %s: %s", logPrefix, errTokenHandlingFails, err.Error())
	}

	if token.Error == nil {
		return &token.Token, nil
	}

	switch *token.Error {
	case "authorization_pending":
		return nil, ErrDeviceAuthorizationPending
	case "slow_down":
		return nil, ErrDeviceSlowDown
	case "access_denied":
		return nil, ErrDeviceAccessDenied
	case "code_expired":
		return nil, ErrDeviceCodeExpired
	default:
		return nil, ErrDeviceGeneric
	}
}

// WaitForUserCompletion calls CheckForUserCompletion repeatedly until a token is granted or an error state occurs.
// This prevents the user from looping and checking against 'ErrDeviceAuthorizationPending'.
// Deprecated: use WaitForUserCompletionWithContext() instead.
func WaitForUserCompletion(sender Sender, code *DeviceCode) (*Token, error) {
	return WaitForUserCompletionWithContext(context.Background(), sender, code)
}

// WaitForUserCompletionWithContext calls CheckForUserCompletion repeatedly until a token is granted or an error
// state occurs.  This prevents the user from looping and checking against 'ErrDeviceAuthorizationPending'.
func WaitForUserCompletionWithContext(ctx context.Context, sender Sender, code *DeviceCode) (*Token, error) {
	intervalDuration := time.Duration(*code.Interval) * time.Second
	waitDuration := intervalDuration

	for {
		token, err := CheckForUserCompletionWithContext(ctx, sender, code)

		if err == nil {
			return token, nil
		}

		switch err {
		case ErrDeviceSlowDown:
			waitDuration += waitDuration
		case ErrDeviceAuthorizationPending:
			// noop
		default: // everything else is "fatal" to us
			return nil, err
		}

		if waitDuration > (intervalDuration * 3) {
			return nil, fmt.Errorf("%s Error waiting for user to complete device flow. Server told us to slow_down too much", logPrefix)
		}

		select {
		case <-time.After(waitDuration):
			// noop
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}
