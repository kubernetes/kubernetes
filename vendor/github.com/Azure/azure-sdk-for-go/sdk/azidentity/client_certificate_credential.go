//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package azidentity

import (
	"context"
	"crypto"
	"crypto/x509"
	"encoding/pem"
	"errors"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/confidential"
	"golang.org/x/crypto/pkcs12"
)

const credNameCert = "ClientCertificateCredential"

// ClientCertificateCredentialOptions contains optional parameters for ClientCertificateCredential.
type ClientCertificateCredentialOptions struct {
	azcore.ClientOptions

	// SendCertificateChain controls whether the credential sends the public certificate chain in the x5c
	// header of each token request's JWT. This is required for Subject Name/Issuer (SNI) authentication.
	// Defaults to False.
	SendCertificateChain bool
}

// ClientCertificateCredential authenticates a service principal with a certificate.
type ClientCertificateCredential struct {
	client confidentialClient
}

// NewClientCertificateCredential constructs a ClientCertificateCredential. Pass nil for options to accept defaults.
func NewClientCertificateCredential(tenantID string, clientID string, certs []*x509.Certificate, key crypto.PrivateKey, options *ClientCertificateCredentialOptions) (*ClientCertificateCredential, error) {
	if len(certs) == 0 {
		return nil, errors.New("at least one certificate is required")
	}
	if options == nil {
		options = &ClientCertificateCredentialOptions{}
	}
	cred, err := confidential.NewCredFromCertChain(certs, key)
	if err != nil {
		return nil, err
	}
	var o []confidential.Option
	if options.SendCertificateChain {
		o = append(o, confidential.WithX5C())
	}
	c, err := getConfidentialClient(clientID, tenantID, cred, &options.ClientOptions, o...)
	if err != nil {
		return nil, err
	}
	return &ClientCertificateCredential{client: c}, nil
}

// GetToken requests an access token from Azure Active Directory. This method is called automatically by Azure SDK clients.
func (c *ClientCertificateCredential) GetToken(ctx context.Context, opts policy.TokenRequestOptions) (azcore.AccessToken, error) {
	if len(opts.Scopes) == 0 {
		return azcore.AccessToken{}, errors.New(credNameCert + ": GetToken() requires at least one scope")
	}
	ar, err := c.client.AcquireTokenSilent(ctx, opts.Scopes)
	if err == nil {
		logGetTokenSuccess(c, opts)
		return azcore.AccessToken{Token: ar.AccessToken, ExpiresOn: ar.ExpiresOn.UTC()}, err
	}

	ar, err = c.client.AcquireTokenByCredential(ctx, opts.Scopes)
	if err != nil {
		return azcore.AccessToken{}, newAuthenticationFailedErrorFromMSALError(credNameCert, err)
	}
	logGetTokenSuccess(c, opts)
	return azcore.AccessToken{Token: ar.AccessToken, ExpiresOn: ar.ExpiresOn.UTC()}, err
}

// ParseCertificates loads certificates and a private key, in PEM or PKCS12 format, for use with NewClientCertificateCredential.
// Pass nil for password if the private key isn't encrypted. This function can't decrypt keys in PEM format.
func ParseCertificates(certData []byte, password []byte) ([]*x509.Certificate, crypto.PrivateKey, error) {
	var blocks []*pem.Block
	var err error
	if len(password) == 0 {
		blocks, err = loadPEMCert(certData)
	}
	if len(blocks) == 0 || err != nil {
		blocks, err = loadPKCS12Cert(certData, string(password))
	}
	if err != nil {
		return nil, nil, err
	}
	var certs []*x509.Certificate
	var pk crypto.PrivateKey
	for _, block := range blocks {
		switch block.Type {
		case "CERTIFICATE":
			c, err := x509.ParseCertificate(block.Bytes)
			if err != nil {
				return nil, nil, err
			}
			certs = append(certs, c)
		case "PRIVATE KEY":
			if pk != nil {
				return nil, nil, errors.New("certData contains multiple private keys")
			}
			pk, err = x509.ParsePKCS8PrivateKey(block.Bytes)
			if err != nil {
				pk, err = x509.ParsePKCS1PrivateKey(block.Bytes)
			}
			if err != nil {
				return nil, nil, err
			}
		case "RSA PRIVATE KEY":
			if pk != nil {
				return nil, nil, errors.New("certData contains multiple private keys")
			}
			pk, err = x509.ParsePKCS1PrivateKey(block.Bytes)
			if err != nil {
				return nil, nil, err
			}
		}
	}
	if len(certs) == 0 {
		return nil, nil, errors.New("found no certificate")
	}
	if pk == nil {
		return nil, nil, errors.New("found no private key")
	}
	return certs, pk, nil
}

func loadPEMCert(certData []byte) ([]*pem.Block, error) {
	blocks := []*pem.Block{}
	for {
		var block *pem.Block
		block, certData = pem.Decode(certData)
		if block == nil {
			break
		}
		blocks = append(blocks, block)
	}
	if len(blocks) == 0 {
		return nil, errors.New("didn't find any PEM blocks")
	}
	return blocks, nil
}

func loadPKCS12Cert(certData []byte, password string) ([]*pem.Block, error) {
	blocks, err := pkcs12.ToPEM(certData, password)
	if err != nil {
		return nil, err
	}
	if len(blocks) == 0 {
		// not mentioning PKCS12 in this message because we end up here when certData is garbage
		return nil, errors.New("didn't find any certificate content")
	}
	return blocks, err
}

var _ azcore.TokenCredential = (*ClientCertificateCredential)(nil)
