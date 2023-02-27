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
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/public"
)

const credNameUserPassword = "UsernamePasswordCredential"

// UsernamePasswordCredentialOptions contains optional parameters for UsernamePasswordCredential.
type UsernamePasswordCredentialOptions struct {
	azcore.ClientOptions
}

// UsernamePasswordCredential authenticates a user with a password. Microsoft doesn't recommend this kind of authentication,
// because it's less secure than other authentication flows. This credential is not interactive, so it isn't compatible
// with any form of multi-factor authentication, and the application must already have user or admin consent.
// This credential can only authenticate work and school accounts; it can't authenticate Microsoft accounts.
type UsernamePasswordCredential struct {
	client   publicClient
	username string
	password string
	account  public.Account
}

// NewUsernamePasswordCredential creates a UsernamePasswordCredential. clientID is the ID of the application the user
// will authenticate to. Pass nil for options to accept defaults.
func NewUsernamePasswordCredential(tenantID string, clientID string, username string, password string, options *UsernamePasswordCredentialOptions) (*UsernamePasswordCredential, error) {
	if options == nil {
		options = &UsernamePasswordCredentialOptions{}
	}
	c, err := getPublicClient(clientID, tenantID, &options.ClientOptions)
	if err != nil {
		return nil, err
	}
	return &UsernamePasswordCredential{username: username, password: password, client: c}, nil
}

// GetToken requests an access token from Azure Active Directory. This method is called automatically by Azure SDK clients.
func (c *UsernamePasswordCredential) GetToken(ctx context.Context, opts policy.TokenRequestOptions) (azcore.AccessToken, error) {
	if len(opts.Scopes) == 0 {
		return azcore.AccessToken{}, errors.New(credNameUserPassword + ": GetToken() requires at least one scope")
	}
	ar, err := c.client.AcquireTokenSilent(ctx, opts.Scopes, public.WithSilentAccount(c.account))
	if err == nil {
		logGetTokenSuccess(c, opts)
		return azcore.AccessToken{Token: ar.AccessToken, ExpiresOn: ar.ExpiresOn.UTC()}, err
	}
	ar, err = c.client.AcquireTokenByUsernamePassword(ctx, opts.Scopes, c.username, c.password)
	if err != nil {
		return azcore.AccessToken{}, newAuthenticationFailedErrorFromMSALError(credNameUserPassword, err)
	}
	c.account = ar.Account
	logGetTokenSuccess(c, opts)
	return azcore.AccessToken{Token: ar.AccessToken, ExpiresOn: ar.ExpiresOn.UTC()}, err
}

var _ azcore.TokenCredential = (*UsernamePasswordCredential)(nil)
