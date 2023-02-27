//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package azidentity

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/internal/log"
)

const envVarSendCertChain = "AZURE_CLIENT_SEND_CERTIFICATE_CHAIN"

// EnvironmentCredentialOptions contains optional parameters for EnvironmentCredential
type EnvironmentCredentialOptions struct {
	azcore.ClientOptions
}

// EnvironmentCredential authenticates a service principal with a secret or certificate, or a user with a password, depending
// on environment variable configuration. It reads configuration from these variables, in the following order:
//
// # Service principal with client secret
//
// AZURE_TENANT_ID: ID of the service principal's tenant. Also called its "directory" ID.
//
// AZURE_CLIENT_ID: the service principal's client ID
//
// AZURE_CLIENT_SECRET: one of the service principal's client secrets
//
// # Service principal with certificate
//
// AZURE_TENANT_ID: ID of the service principal's tenant. Also called its "directory" ID.
//
// AZURE_CLIENT_ID: the service principal's client ID
//
// AZURE_CLIENT_CERTIFICATE_PATH: path to a PEM or PKCS12 certificate file including the private key.
//
// AZURE_CLIENT_CERTIFICATE_PASSWORD: (optional) password for the certificate file.
//
// # User with username and password
//
// AZURE_TENANT_ID: (optional) tenant to authenticate in. Defaults to "organizations".
//
// AZURE_CLIENT_ID: client ID of the application the user will authenticate to
//
// AZURE_USERNAME: a username (usually an email address)
//
// AZURE_PASSWORD: the user's password
type EnvironmentCredential struct {
	cred azcore.TokenCredential
}

// NewEnvironmentCredential creates an EnvironmentCredential. Pass nil to accept default options.
func NewEnvironmentCredential(options *EnvironmentCredentialOptions) (*EnvironmentCredential, error) {
	if options == nil {
		options = &EnvironmentCredentialOptions{}
	}
	tenantID := os.Getenv(azureTenantID)
	if tenantID == "" {
		return nil, errors.New("missing environment variable AZURE_TENANT_ID")
	}
	clientID := os.Getenv(azureClientID)
	if clientID == "" {
		return nil, errors.New("missing environment variable " + azureClientID)
	}
	if clientSecret := os.Getenv(azureClientSecret); clientSecret != "" {
		log.Write(EventAuthentication, "EnvironmentCredential will authenticate with ClientSecretCredential")
		o := &ClientSecretCredentialOptions{ClientOptions: options.ClientOptions}
		cred, err := NewClientSecretCredential(tenantID, clientID, clientSecret, o)
		if err != nil {
			return nil, err
		}
		return &EnvironmentCredential{cred: cred}, nil
	}
	if certPath := os.Getenv(azureClientCertificatePath); certPath != "" {
		log.Write(EventAuthentication, "EnvironmentCredential will authenticate with ClientCertificateCredential")
		certData, err := os.ReadFile(certPath)
		if err != nil {
			return nil, fmt.Errorf(`failed to read certificate file "%s": %v`, certPath, err)
		}
		var password []byte
		if v := os.Getenv(azureClientCertificatePassword); v != "" {
			password = []byte(v)
		}
		certs, key, err := ParseCertificates(certData, password)
		if err != nil {
			return nil, fmt.Errorf(`failed to load certificate from "%s": %v`, certPath, err)
		}
		o := &ClientCertificateCredentialOptions{ClientOptions: options.ClientOptions}
		if v, ok := os.LookupEnv(envVarSendCertChain); ok {
			o.SendCertificateChain = v == "1" || strings.ToLower(v) == "true"
		}
		cred, err := NewClientCertificateCredential(tenantID, clientID, certs, key, o)
		if err != nil {
			return nil, err
		}
		return &EnvironmentCredential{cred: cred}, nil
	}
	if username := os.Getenv(azureUsername); username != "" {
		if password := os.Getenv(azurePassword); password != "" {
			log.Write(EventAuthentication, "EnvironmentCredential will authenticate with UsernamePasswordCredential")
			o := &UsernamePasswordCredentialOptions{ClientOptions: options.ClientOptions}
			cred, err := NewUsernamePasswordCredential(tenantID, clientID, username, password, o)
			if err != nil {
				return nil, err
			}
			return &EnvironmentCredential{cred: cred}, nil
		}
		return nil, errors.New("no value for AZURE_PASSWORD")
	}
	return nil, errors.New("incomplete environment variable configuration. Only AZURE_TENANT_ID and AZURE_CLIENT_ID are set")
}

// GetToken requests an access token from Azure Active Directory. This method is called automatically by Azure SDK clients.
func (c *EnvironmentCredential) GetToken(ctx context.Context, opts policy.TokenRequestOptions) (azcore.AccessToken, error) {
	return c.cred.GetToken(ctx, opts)
}

var _ azcore.TokenCredential = (*EnvironmentCredential)(nil)
