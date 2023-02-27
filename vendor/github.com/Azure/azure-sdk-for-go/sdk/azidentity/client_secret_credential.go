//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package azidentity

import (
	"context"
	"errors"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/confidential"
)

const credNameSecret = "ClientSecretCredential"

// ClientSecretCredentialOptions contains optional parameters for ClientSecretCredential.
type ClientSecretCredentialOptions struct {
	azcore.ClientOptions
}

// ClientSecretCredential authenticates an application with a client secret.
type ClientSecretCredential struct {
	client confidentialClient
}

// NewClientSecretCredential constructs a ClientSecretCredential. Pass nil for options to accept defaults.
func NewClientSecretCredential(tenantID string, clientID string, clientSecret string, options *ClientSecretCredentialOptions) (*ClientSecretCredential, error) {
	if options == nil {
		options = &ClientSecretCredentialOptions{}
	}
	cred, err := confidential.NewCredFromSecret(clientSecret)
	if err != nil {
		return nil, err
	}
	c, err := getConfidentialClient(clientID, tenantID, cred, &options.ClientOptions)
	if err != nil {
		return nil, err
	}
	return &ClientSecretCredential{client: c}, nil
}

// GetToken requests an access token from Azure Active Directory. This method is called automatically by Azure SDK clients.
func (c *ClientSecretCredential) GetToken(ctx context.Context, opts policy.TokenRequestOptions) (azcore.AccessToken, error) {
	if len(opts.Scopes) == 0 {
		return azcore.AccessToken{}, errors.New(credNameSecret + ": GetToken() requires at least one scope")
	}
	ar, err := c.client.AcquireTokenSilent(ctx, opts.Scopes)
	if err == nil {
		logGetTokenSuccess(c, opts)
		return azcore.AccessToken{Token: ar.AccessToken, ExpiresOn: ar.ExpiresOn.UTC()}, err
	}

	ar, err = c.client.AcquireTokenByCredential(ctx, opts.Scopes)
	if err != nil {
		return azcore.AccessToken{}, newAuthenticationFailedErrorFromMSALError(credNameSecret, err)
	}
	logGetTokenSuccess(c, opts)
	return azcore.AccessToken{Token: ar.AccessToken, ExpiresOn: ar.ExpiresOn.UTC()}, err
}

var _ azcore.TokenCredential = (*ClientSecretCredential)(nil)
