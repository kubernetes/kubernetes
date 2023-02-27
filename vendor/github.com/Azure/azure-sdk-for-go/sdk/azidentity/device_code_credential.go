//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package azidentity

import (
	"context"
	"errors"
	"fmt"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/public"
)

const credNameDeviceCode = "DeviceCodeCredential"

// DeviceCodeCredentialOptions contains optional parameters for DeviceCodeCredential.
type DeviceCodeCredentialOptions struct {
	azcore.ClientOptions

	// TenantID is the Azure Active Directory tenant the credential authenticates in. Defaults to the
	// "organizations" tenant, which can authenticate work and school accounts. Required for single-tenant
	// applications.
	TenantID string
	// ClientID is the ID of the application users will authenticate to.
	// Defaults to the ID of an Azure development application.
	ClientID string
	// UserPrompt controls how the credential presents authentication instructions. The credential calls
	// this function with authentication details when it receives a device code. By default, the credential
	// prints these details to stdout.
	UserPrompt func(context.Context, DeviceCodeMessage) error
}

func (o *DeviceCodeCredentialOptions) init() {
	if o.TenantID == "" {
		o.TenantID = organizationsTenantID
	}
	if o.ClientID == "" {
		o.ClientID = developerSignOnClientID
	}
	if o.UserPrompt == nil {
		o.UserPrompt = func(ctx context.Context, dc DeviceCodeMessage) error {
			fmt.Println(dc.Message)
			return nil
		}
	}
}

// DeviceCodeMessage contains the information a user needs to complete authentication.
type DeviceCodeMessage struct {
	// UserCode is the user code returned by the service.
	UserCode string `json:"user_code"`
	// VerificationURL is the URL at which the user must authenticate.
	VerificationURL string `json:"verification_uri"`
	// Message is user instruction from Azure Active Directory.
	Message string `json:"message"`
}

// DeviceCodeCredential acquires tokens for a user via the device code flow, which has the
// user browse to an Azure Active Directory URL, enter a code, and authenticate. It's useful
// for authenticating a user in an environment without a web browser, such as an SSH session.
// If a web browser is available, InteractiveBrowserCredential is more convenient because it
// automatically opens a browser to the login page.
type DeviceCodeCredential struct {
	client     publicClient
	userPrompt func(context.Context, DeviceCodeMessage) error
	account    public.Account
}

// NewDeviceCodeCredential creates a DeviceCodeCredential. Pass nil to accept default options.
func NewDeviceCodeCredential(options *DeviceCodeCredentialOptions) (*DeviceCodeCredential, error) {
	cp := DeviceCodeCredentialOptions{}
	if options != nil {
		cp = *options
	}
	cp.init()
	c, err := getPublicClient(cp.ClientID, cp.TenantID, &cp.ClientOptions)
	if err != nil {
		return nil, err
	}
	return &DeviceCodeCredential{userPrompt: cp.UserPrompt, client: c}, nil
}

// GetToken requests an access token from Azure Active Directory. It will begin the device code flow and poll until the user completes authentication.
// This method is called automatically by Azure SDK clients.
func (c *DeviceCodeCredential) GetToken(ctx context.Context, opts policy.TokenRequestOptions) (azcore.AccessToken, error) {
	if len(opts.Scopes) == 0 {
		return azcore.AccessToken{}, errors.New(credNameDeviceCode + ": GetToken() requires at least one scope")
	}
	ar, err := c.client.AcquireTokenSilent(ctx, opts.Scopes, public.WithSilentAccount(c.account))
	if err == nil {
		return azcore.AccessToken{Token: ar.AccessToken, ExpiresOn: ar.ExpiresOn.UTC()}, err
	}
	dc, err := c.client.AcquireTokenByDeviceCode(ctx, opts.Scopes)
	if err != nil {
		return azcore.AccessToken{}, newAuthenticationFailedErrorFromMSALError(credNameDeviceCode, err)
	}
	err = c.userPrompt(ctx, DeviceCodeMessage{
		UserCode:        dc.Result.UserCode,
		VerificationURL: dc.Result.VerificationURL,
		Message:         dc.Result.Message,
	})
	if err != nil {
		return azcore.AccessToken{}, err
	}
	ar, err = dc.AuthenticationResult(ctx)
	if err != nil {
		return azcore.AccessToken{}, newAuthenticationFailedErrorFromMSALError(credNameDeviceCode, err)
	}
	c.account = ar.Account
	logGetTokenSuccess(c, opts)
	return azcore.AccessToken{Token: ar.AccessToken, ExpiresOn: ar.ExpiresOn.UTC()}, err
}

var _ azcore.TokenCredential = (*DeviceCodeCredential)(nil)
