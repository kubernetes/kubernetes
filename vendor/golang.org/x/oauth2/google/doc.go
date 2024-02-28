// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package google provides support for making OAuth2 authorized and authenticated
// HTTP requests to Google APIs. It supports the Web server flow, client-side
// credentials, service accounts, Google Compute Engine service accounts,
// Google App Engine service accounts and workload identity federation
// from non-Google cloud platforms.
//
// A brief overview of the package follows. For more information, please read
// https://developers.google.com/accounts/docs/OAuth2
// and
// https://developers.google.com/accounts/docs/application-default-credentials.
// For more information on using workload identity federation, refer to
// https://cloud.google.com/iam/docs/how-to#using-workload-identity-federation.
//
// # OAuth2 Configs
//
// Two functions in this package return golang.org/x/oauth2.Config values from Google credential
// data. Google supports two JSON formats for OAuth2 credentials: one is handled by ConfigFromJSON,
// the other by JWTConfigFromJSON. The returned Config can be used to obtain a TokenSource or
// create an http.Client.
//
// # Workload Identity Federation
//
// Using workload identity federation, your application can access Google Cloud
// resources from Amazon Web Services (AWS), Microsoft Azure or any identity
// provider that supports OpenID Connect (OIDC) or SAML 2.0.
// Traditionally, applications running outside Google Cloud have used service
// account keys to access Google Cloud resources. Using identity federation,
// you can allow your workload to impersonate a service account.
// This lets you access Google Cloud resources directly, eliminating the
// maintenance and security burden associated with service account keys.
//
// Follow the detailed instructions on how to configure Workload Identity Federation
// in various platforms:
//
//	Amazon Web Services (AWS): https://cloud.google.com/iam/docs/workload-identity-federation-with-other-clouds#aws
//	Microsoft Azure: https://cloud.google.com/iam/docs/workload-identity-federation-with-other-clouds#azure
//	OIDC identity provider: https://cloud.google.com/iam/docs/workload-identity-federation-with-other-providers#oidc
//	SAML 2.0 identity provider: https://cloud.google.com/iam/docs/workload-identity-federation-with-other-providers#saml
//
// For OIDC and SAML providers, the library can retrieve tokens in three ways:
// from a local file location (file-sourced credentials), from a server
// (URL-sourced credentials), or from a local executable (executable-sourced
// credentials).
// For file-sourced credentials, a background process needs to be continuously
// refreshing the file location with a new OIDC/SAML token prior to expiration.
// For tokens with one hour lifetimes, the token needs to be updated in the file
// every hour. The token can be stored directly as plain text or in JSON format.
// For URL-sourced credentials, a local server needs to host a GET endpoint to
// return the OIDC/SAML token. The response can be in plain text or JSON.
// Additional required request headers can also be specified.
// For executable-sourced credentials, an application needs to be available to
// output the OIDC/SAML token and other information in a JSON format.
// For more information on how these work (and how to implement
// executable-sourced credentials), please check out:
// https://cloud.google.com/iam/docs/workload-identity-federation-with-other-providers#create_a_credential_configuration
//
// Note that this library does not perform any validation on the token_url, token_info_url,
// or service_account_impersonation_url fields of the credential configuration.
// It is not recommended to use a credential configuration that you did not generate with
// the gcloud CLI unless you verify that the URL fields point to a googleapis.com domain.
//
// # Workforce Identity Federation
//
// Workforce identity federation lets you use an external identity provider (IdP) to
// authenticate and authorize a workforce—a group of users, such as employees, partners,
// and contractors—using IAM, so that the users can access Google Cloud services.
// Workforce identity federation extends Google Cloud's identity capabilities to support
// syncless, attribute-based single sign on.
//
// With workforce identity federation, your workforce can access Google Cloud resources
// using an external identity provider (IdP) that supports OpenID Connect (OIDC) or
// SAML 2.0 such as Azure Active Directory (Azure AD), Active Directory Federation
// Services (AD FS), Okta, and others.
//
// Follow the detailed instructions on how to configure Workload Identity Federation
// in various platforms:
//
//	Azure AD: https://cloud.google.com/iam/docs/workforce-sign-in-azure-ad
//	Okta: https://cloud.google.com/iam/docs/workforce-sign-in-okta
//	OIDC identity provider: https://cloud.google.com/iam/docs/configuring-workforce-identity-federation#oidc
//	SAML 2.0 identity provider: https://cloud.google.com/iam/docs/configuring-workforce-identity-federation#saml
//
// For workforce identity federation, the library can retrieve tokens in three ways:
// from a local file location (file-sourced credentials), from a server
// (URL-sourced credentials), or from a local executable (executable-sourced
// credentials).
// For file-sourced credentials, a background process needs to be continuously
// refreshing the file location with a new OIDC/SAML token prior to expiration.
// For tokens with one hour lifetimes, the token needs to be updated in the file
// every hour. The token can be stored directly as plain text or in JSON format.
// For URL-sourced credentials, a local server needs to host a GET endpoint to
// return the OIDC/SAML token. The response can be in plain text or JSON.
// Additional required request headers can also be specified.
// For executable-sourced credentials, an application needs to be available to
// output the OIDC/SAML token and other information in a JSON format.
// For more information on how these work (and how to implement
// executable-sourced credentials), please check out:
// https://cloud.google.com/iam/docs/workforce-obtaining-short-lived-credentials#generate_a_configuration_file_for_non-interactive_sign-in
//
// # Security considerations
//
// Note that this library does not perform any validation on the token_url, token_info_url,
// or service_account_impersonation_url fields of the credential configuration.
// It is not recommended to use a credential configuration that you did not generate with
// the gcloud CLI unless you verify that the URL fields point to a googleapis.com domain.
//
// # Credentials
//
// The Credentials type represents Google credentials, including Application Default
// Credentials.
//
// Use FindDefaultCredentials to obtain Application Default Credentials.
// FindDefaultCredentials looks in some well-known places for a credentials file, and
// will call AppEngineTokenSource or ComputeTokenSource as needed.
//
// Application Default Credentials also support workload identity federation to
// access Google Cloud resources from non-Google Cloud platforms including Amazon
// Web Services (AWS), Microsoft Azure or any identity provider that supports
// OpenID Connect (OIDC). Workload identity federation is recommended for
// non-Google Cloud environments as it avoids the need to download, manage and
// store service account private keys locally.
//
// DefaultClient and DefaultTokenSource are convenience methods. They first call FindDefaultCredentials,
// then use the credentials to construct an http.Client or an oauth2.TokenSource.
//
// Use CredentialsFromJSON to obtain credentials from either of the two JSON formats
// described in OAuth2 Configs, above. The TokenSource in the returned value is the
// same as the one obtained from the oauth2.Config returned from ConfigFromJSON or
// JWTConfigFromJSON, but the Credentials may contain additional information
// that is useful is some circumstances.
package google // import "golang.org/x/oauth2/google"
