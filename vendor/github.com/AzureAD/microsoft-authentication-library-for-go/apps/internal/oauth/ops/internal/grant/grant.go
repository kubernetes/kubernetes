// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Package grant holds types of grants issued by authorization services.
package grant

const (
	Password         = "password"
	JWT              = "urn:ietf:params:oauth:grant-type:jwt-bearer"
	SAMLV1           = "urn:ietf:params:oauth:grant-type:saml1_1-bearer"
	SAMLV2           = "urn:ietf:params:oauth:grant-type:saml2-bearer"
	DeviceCode       = "device_code"
	AuthCode         = "authorization_code"
	RefreshToken     = "refresh_token"
	ClientCredential = "client_credentials"
	ClientAssertion  = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
)
