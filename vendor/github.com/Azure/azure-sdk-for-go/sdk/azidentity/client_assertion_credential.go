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

const credNameAssertion = "ClientAssertionCredential"

// ClientAssertionCredential authenticates an application with assertions provided by a callback function.
// This credential is for advanced scenarios. ClientCertificateCredential has a more convenient API for
// the most common assertion scenario, authenticating a service principal with a certificate. See
// [Azure AD documentation] for details of the assertion format.
//
// [Azure AD documentation]: https://docs.microsoft.com/azure/active-directory/develop/active-directory-certificate-credentials#assertion-format
type ClientAssertionCredential struct {
	client confidentialClient
}

// ClientAssertionCredentialOptions contains optional parameters for ClientAssertionCredential.
type ClientAssertionCredentialOptions struct {
	azcore.ClientOptions
}

// NewClientAssertionCredential constructs a ClientAssertionCredential. The getAssertion function must be thread safe. Pass nil for options to accept defaults.
func NewClientAssertionCredential(tenantID, clientID string, getAssertion func(context.Context) (string, error), options *ClientAssertionCredentialOptions) (*ClientAssertionCredential, error) {
	if getAssertion == nil {
		return nil, errors.New("getAssertion must be a function that returns assertions")
	}
	if options == nil {
		options = &ClientAssertionCredentialOptions{}
	}
	cred := confidential.NewCredFromAssertionCallback(
		func(ctx context.Context, _ confidential.AssertionRequestOptions) (string, error) {
			return getAssertion(ctx)
		},
	)
	c, err := getConfidentialClient(clientID, tenantID, cred, &options.ClientOptions)
	if err != nil {
		return nil, err
	}
	return &ClientAssertionCredential{client: c}, nil
}

// GetToken requests an access token from Azure Active Directory. This method is called automatically by Azure SDK clients.
func (c *ClientAssertionCredential) GetToken(ctx context.Context, opts policy.TokenRequestOptions) (azcore.AccessToken, error) {
	if len(opts.Scopes) == 0 {
		return azcore.AccessToken{}, errors.New(credNameAssertion + ": GetToken() requires at least one scope")
	}
	ar, err := c.client.AcquireTokenSilent(ctx, opts.Scopes)
	if err == nil {
		logGetTokenSuccess(c, opts)
		return azcore.AccessToken{Token: ar.AccessToken, ExpiresOn: ar.ExpiresOn.UTC()}, err
	}

	ar, err = c.client.AcquireTokenByCredential(ctx, opts.Scopes)
	if err != nil {
		return azcore.AccessToken{}, newAuthenticationFailedErrorFromMSALError(credNameAssertion, err)
	}
	logGetTokenSuccess(c, opts)
	return azcore.AccessToken{Token: ar.AccessToken, ExpiresOn: ar.ExpiresOn.UTC()}, err
}

var _ azcore.TokenCredential = (*ClientAssertionCredential)(nil)
