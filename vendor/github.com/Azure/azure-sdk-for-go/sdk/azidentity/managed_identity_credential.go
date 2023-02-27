//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package azidentity

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/confidential"
)

const credNameManagedIdentity = "ManagedIdentityCredential"

type managedIdentityIDKind int

const (
	miClientID   managedIdentityIDKind = 0
	miResourceID managedIdentityIDKind = 1
)

// ManagedIDKind identifies the ID of a managed identity as either a client or resource ID
type ManagedIDKind interface {
	fmt.Stringer
	idKind() managedIdentityIDKind
}

// ClientID is the client ID of a user-assigned managed identity.
type ClientID string

func (ClientID) idKind() managedIdentityIDKind {
	return miClientID
}

// String returns the string value of the ID.
func (c ClientID) String() string {
	return string(c)
}

// ResourceID is the resource ID of a user-assigned managed identity.
type ResourceID string

func (ResourceID) idKind() managedIdentityIDKind {
	return miResourceID
}

// String returns the string value of the ID.
func (r ResourceID) String() string {
	return string(r)
}

// ManagedIdentityCredentialOptions contains optional parameters for ManagedIdentityCredential.
type ManagedIdentityCredentialOptions struct {
	azcore.ClientOptions

	// ID is the ID of a managed identity the credential should authenticate. Set this field to use a specific identity
	// instead of the hosting environment's default. The value may be the identity's client ID or resource ID, but note that
	// some platforms don't accept resource IDs.
	ID ManagedIDKind
}

// ManagedIdentityCredential authenticates an Azure managed identity in any hosting environment supporting managed identities.
// This credential authenticates a system-assigned identity by default. Use ManagedIdentityCredentialOptions.ID to specify a
// user-assigned identity. See Azure Active Directory documentation for more information about managed identities:
// https://docs.microsoft.com/azure/active-directory/managed-identities-azure-resources/overview
type ManagedIdentityCredential struct {
	client confidentialClient
	mic    *managedIdentityClient
}

// NewManagedIdentityCredential creates a ManagedIdentityCredential. Pass nil to accept default options.
func NewManagedIdentityCredential(options *ManagedIdentityCredentialOptions) (*ManagedIdentityCredential, error) {
	if options == nil {
		options = &ManagedIdentityCredentialOptions{}
	}
	mic, err := newManagedIdentityClient(options)
	if err != nil {
		return nil, err
	}
	cred := confidential.NewCredFromTokenProvider(mic.provideToken)
	if err != nil {
		return nil, err
	}
	// It's okay to give MSAL an invalid client ID because MSAL will use it only as part of a cache key.
	// ManagedIdentityClient handles all the details of authentication and won't receive this value from MSAL.
	clientID := "SYSTEM-ASSIGNED-MANAGED-IDENTITY"
	if options.ID != nil {
		clientID = options.ID.String()
	}
	c, err := confidential.New(clientID, cred)
	if err != nil {
		return nil, err
	}
	return &ManagedIdentityCredential{client: c, mic: mic}, nil
}

// GetToken requests an access token from the hosting environment. This method is called automatically by Azure SDK clients.
func (c *ManagedIdentityCredential) GetToken(ctx context.Context, opts policy.TokenRequestOptions) (azcore.AccessToken, error) {
	if len(opts.Scopes) != 1 {
		err := errors.New(credNameManagedIdentity + ": GetToken() requires exactly one scope")
		return azcore.AccessToken{}, err
	}
	// managed identity endpoints require an AADv1 resource (i.e. token audience), not a v2 scope, so we remove "/.default" here
	scopes := []string{strings.TrimSuffix(opts.Scopes[0], defaultSuffix)}
	ar, err := c.client.AcquireTokenSilent(ctx, scopes)
	if err == nil {
		logGetTokenSuccess(c, opts)
		return azcore.AccessToken{Token: ar.AccessToken, ExpiresOn: ar.ExpiresOn.UTC()}, nil
	}
	ar, err = c.client.AcquireTokenByCredential(ctx, scopes)
	if err != nil {
		return azcore.AccessToken{}, err
	}
	logGetTokenSuccess(c, opts)
	return azcore.AccessToken{Token: ar.AccessToken, ExpiresOn: ar.ExpiresOn.UTC()}, err
}

var _ azcore.TokenCredential = (*ManagedIdentityCredential)(nil)
