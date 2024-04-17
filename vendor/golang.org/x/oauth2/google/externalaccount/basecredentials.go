// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package externalaccount provides support for creating workload identity
federation and workforce identity federation token sources that can be
used to access Google Cloud resources from external identity providers.

# Workload Identity Federation

Using workload identity federation, your application can access Google Cloud
resources from Amazon Web Services (AWS), Microsoft Azure or any identity
provider that supports OpenID Connect (OIDC) or SAML 2.0.
Traditionally, applications running outside Google Cloud have used service
account keys to access Google Cloud resources. Using identity federation,
you can allow your workload to impersonate a service account.
This lets you access Google Cloud resources directly, eliminating the
maintenance and security burden associated with service account keys.

Follow the detailed instructions on how to configure Workload Identity Federation
in various platforms:

Amazon Web Services (AWS): https://cloud.google.com/iam/docs/workload-identity-federation-with-other-clouds#aws
Microsoft Azure: https://cloud.google.com/iam/docs/workload-identity-federation-with-other-clouds#azure
OIDC identity provider: https://cloud.google.com/iam/docs/workload-identity-federation-with-other-providers#oidc
SAML 2.0 identity provider: https://cloud.google.com/iam/docs/workload-identity-federation-with-other-providers#saml

For OIDC and SAML providers, the library can retrieve tokens in fours ways:
from a local file location (file-sourced credentials), from a server
(URL-sourced credentials), from a local executable (executable-sourced
credentials), or from a user defined function that returns an OIDC or SAML token.
For file-sourced credentials, a background process needs to be continuously
refreshing the file location with a new OIDC/SAML token prior to expiration.
For tokens with one hour lifetimes, the token needs to be updated in the file
every hour. The token can be stored directly as plain text or in JSON format.
For URL-sourced credentials, a local server needs to host a GET endpoint to
return the OIDC/SAML token. The response can be in plain text or JSON.
Additional required request headers can also be specified.
For executable-sourced credentials, an application needs to be available to
output the OIDC/SAML token and other information in a JSON format.
For more information on how these work (and how to implement
executable-sourced credentials), please check out:
https://cloud.google.com/iam/docs/workload-identity-federation-with-other-providers#create_a_credential_configuration

To use a custom function to supply the token, define a struct that implements the [SubjectTokenSupplier] interface for OIDC/SAML providers,
or one that implements [AwsSecurityCredentialsSupplier] for AWS providers. This can then be used when building a [Config].
The [golang.org/x/oauth2.TokenSource] created from the config using [NewTokenSource] can then be used to access Google
Cloud resources. For instance, you can create a new client from the
[cloud.google.com/go/storage] package and pass in option.WithTokenSource(yourTokenSource))

Note that this library does not perform any validation on the token_url, token_info_url,
or service_account_impersonation_url fields of the credential configuration.
It is not recommended to use a credential configuration that you did not generate with
the gcloud CLI unless you verify that the URL fields point to a googleapis.com domain.

# Workforce Identity Federation

Workforce identity federation lets you use an external identity provider (IdP) to
authenticate and authorize a workforce—a group of users, such as employees, partners,
and contractors—using IAM, so that the users can access Google Cloud services.
Workforce identity federation extends Google Cloud's identity capabilities to support
syncless, attribute-based single sign on.

With workforce identity federation, your workforce can access Google Cloud resources
using an external identity provider (IdP) that supports OpenID Connect (OIDC) or
SAML 2.0 such as Azure Active Directory (Azure AD), Active Directory Federation
Services (AD FS), Okta, and others.

Follow the detailed instructions on how to configure Workload Identity Federation
in various platforms:

Azure AD: https://cloud.google.com/iam/docs/workforce-sign-in-azure-ad
Okta: https://cloud.google.com/iam/docs/workforce-sign-in-okta
OIDC identity provider: https://cloud.google.com/iam/docs/configuring-workforce-identity-federation#oidc
SAML 2.0 identity provider: https://cloud.google.com/iam/docs/configuring-workforce-identity-federation#saml

For workforce identity federation, the library can retrieve tokens in four ways:
from a local file location (file-sourced credentials), from a server
(URL-sourced credentials), from a local executable (executable-sourced
credentials), or from a user supplied function that returns an OIDC or SAML token.
For file-sourced credentials, a background process needs to be continuously
refreshing the file location with a new OIDC/SAML token prior to expiration.
For tokens with one hour lifetimes, the token needs to be updated in the file
every hour. The token can be stored directly as plain text or in JSON format.
For URL-sourced credentials, a local server needs to host a GET endpoint to
return the OIDC/SAML token. The response can be in plain text or JSON.
Additional required request headers can also be specified.
For executable-sourced credentials, an application needs to be available to
output the OIDC/SAML token and other information in a JSON format.
For more information on how these work (and how to implement
executable-sourced credentials), please check out:
https://cloud.google.com/iam/docs/workforce-obtaining-short-lived-credentials#generate_a_configuration_file_for_non-interactive_sign-in

To use a custom function to supply the token, define a struct that implements the [SubjectTokenSupplier] interface for OIDC/SAML providers.
This can then be used when building a [Config].
The [golang.org/x/oauth2.TokenSource] created from the config using [NewTokenSource] can then be used access Google
Cloud resources. For instance, you can create a new client from the
[cloud.google.com/go/storage] package and pass in option.WithTokenSource(yourTokenSource))

# Security considerations

Note that this library does not perform any validation on the token_url, token_info_url,
or service_account_impersonation_url fields of the credential configuration.
It is not recommended to use a credential configuration that you did not generate with
the gcloud CLI unless you verify that the URL fields point to a googleapis.com domain.
*/
package externalaccount

import (
	"context"
	"fmt"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google/internal/impersonate"
	"golang.org/x/oauth2/google/internal/stsexchange"
)

const (
	universeDomainPlaceholder = "UNIVERSE_DOMAIN"
	defaultTokenURL           = "https://sts.UNIVERSE_DOMAIN/v1/token"
	defaultUniverseDomain     = "googleapis.com"
)

// now aliases time.Now for testing
var now = func() time.Time {
	return time.Now().UTC()
}

// Config stores the configuration for fetching tokens with external credentials.
type Config struct {
	// Audience is the Secure Token Service (STS) audience which contains the resource name for the workload
	// identity pool or the workforce pool and the provider identifier in that pool. Required.
	Audience string
	// SubjectTokenType is the STS token type based on the Oauth2.0 token exchange spec.
	// Expected values include:
	// “urn:ietf:params:oauth:token-type:jwt”
	// “urn:ietf:params:oauth:token-type:id-token”
	// “urn:ietf:params:oauth:token-type:saml2”
	// “urn:ietf:params:aws:token-type:aws4_request”
	// Required.
	SubjectTokenType string
	// TokenURL is the STS token exchange endpoint. If not provided, will default to
	// https://sts.UNIVERSE_DOMAIN/v1/token, with UNIVERSE_DOMAIN set to the
	// default service domain googleapis.com unless UniverseDomain is set.
	// Optional.
	TokenURL string
	// TokenInfoURL is the token_info endpoint used to retrieve the account related information (
	// user attributes like account identifier, eg. email, username, uid, etc). This is
	// needed for gCloud session account identification. Optional.
	TokenInfoURL string
	// ServiceAccountImpersonationURL is the URL for the service account impersonation request. This is only
	// required for workload identity pools when APIs to be accessed have not integrated with UberMint. Optional.
	ServiceAccountImpersonationURL string
	// ServiceAccountImpersonationLifetimeSeconds is the number of seconds the service account impersonation
	// token will be valid for. If not provided, it will default to 3600. Optional.
	ServiceAccountImpersonationLifetimeSeconds int
	// ClientSecret is currently only required if token_info endpoint also
	// needs to be called with the generated GCP access token. When provided, STS will be
	// called with additional basic authentication using ClientId as username and ClientSecret as password. Optional.
	ClientSecret string
	// ClientID is only required in conjunction with ClientSecret, as described above. Optional.
	ClientID string
	// CredentialSource contains the necessary information to retrieve the token itself, as well
	// as some environmental information. One of SubjectTokenSupplier, AWSSecurityCredentialSupplier or
	// CredentialSource must be provided. Optional.
	CredentialSource *CredentialSource
	// QuotaProjectID is injected by gCloud. If the value is non-empty, the Auth libraries
	// will set the x-goog-user-project header which overrides the project associated with the credentials. Optional.
	QuotaProjectID string
	// Scopes contains the desired scopes for the returned access token. Optional.
	Scopes []string
	// WorkforcePoolUserProject is the workforce pool user project number when the credential
	// corresponds to a workforce pool and not a workload identity pool.
	// The underlying principal must still have serviceusage.services.use IAM
	// permission to use the project for billing/quota. Optional.
	WorkforcePoolUserProject string
	// SubjectTokenSupplier is an optional token supplier for OIDC/SAML credentials.
	// One of SubjectTokenSupplier, AWSSecurityCredentialSupplier or CredentialSource must be provided. Optional.
	SubjectTokenSupplier SubjectTokenSupplier
	// AwsSecurityCredentialsSupplier is an AWS Security Credential supplier for AWS credentials.
	// One of SubjectTokenSupplier, AWSSecurityCredentialSupplier or CredentialSource must be provided. Optional.
	AwsSecurityCredentialsSupplier AwsSecurityCredentialsSupplier
	// UniverseDomain is the default service domain for a given Cloud universe.
	// This value will be used in the default STS token URL. The default value
	// is "googleapis.com". It will not be used if TokenURL is set. Optional.
	UniverseDomain string
}

var (
	validWorkforceAudiencePattern *regexp.Regexp = regexp.MustCompile(`//iam\.googleapis\.com/locations/[^/]+/workforcePools/`)
)

func validateWorkforceAudience(input string) bool {
	return validWorkforceAudiencePattern.MatchString(input)
}

// NewTokenSource Returns an external account TokenSource using the provided external account config.
func NewTokenSource(ctx context.Context, conf Config) (oauth2.TokenSource, error) {
	if conf.Audience == "" {
		return nil, fmt.Errorf("oauth2/google/externalaccount: Audience must be set")
	}
	if conf.SubjectTokenType == "" {
		return nil, fmt.Errorf("oauth2/google/externalaccount: Subject token type must be set")
	}
	if conf.WorkforcePoolUserProject != "" {
		valid := validateWorkforceAudience(conf.Audience)
		if !valid {
			return nil, fmt.Errorf("oauth2/google/externalaccount: Workforce pool user project should not be set for non-workforce pool credentials")
		}
	}
	count := 0
	if conf.CredentialSource != nil {
		count++
	}
	if conf.SubjectTokenSupplier != nil {
		count++
	}
	if conf.AwsSecurityCredentialsSupplier != nil {
		count++
	}
	if count == 0 {
		return nil, fmt.Errorf("oauth2/google/externalaccount: One of CredentialSource, SubjectTokenSupplier, or AwsSecurityCredentialsSupplier must be set")
	}
	if count > 1 {
		return nil, fmt.Errorf("oauth2/google/externalaccount: Only one of CredentialSource, SubjectTokenSupplier, or AwsSecurityCredentialsSupplier must be set")
	}
	return conf.tokenSource(ctx, "https")
}

// tokenSource is a private function that's directly called by some of the tests,
// because the unit test URLs are mocked, and would otherwise fail the
// validity check.
func (c *Config) tokenSource(ctx context.Context, scheme string) (oauth2.TokenSource, error) {

	ts := tokenSource{
		ctx:  ctx,
		conf: c,
	}
	if c.ServiceAccountImpersonationURL == "" {
		return oauth2.ReuseTokenSource(nil, ts), nil
	}
	scopes := c.Scopes
	ts.conf.Scopes = []string{"https://www.googleapis.com/auth/cloud-platform"}
	imp := impersonate.ImpersonateTokenSource{
		Ctx:                  ctx,
		URL:                  c.ServiceAccountImpersonationURL,
		Scopes:               scopes,
		Ts:                   oauth2.ReuseTokenSource(nil, ts),
		TokenLifetimeSeconds: c.ServiceAccountImpersonationLifetimeSeconds,
	}
	return oauth2.ReuseTokenSource(nil, imp), nil
}

// Subject token file types.
const (
	fileTypeText = "text"
	fileTypeJSON = "json"
)

// Format contains information needed to retireve a subject token for URL or File sourced credentials.
type Format struct {
	// Type should be either "text" or "json". This determines whether the file or URL sourced credentials
	// expect a simple text subject token or if the subject token will be contained in a JSON object.
	// When not provided "text" type is assumed.
	Type string `json:"type"`
	// SubjectTokenFieldName is only required for JSON format. This is the field name that the credentials will check
	// for the subject token in the file or URL response. This would be "access_token" for azure.
	SubjectTokenFieldName string `json:"subject_token_field_name"`
}

// CredentialSource stores the information necessary to retrieve the credentials for the STS exchange.
type CredentialSource struct {
	// File is the location for file sourced credentials.
	// One field amongst File, URL, Executable, or EnvironmentID should be provided, depending on the kind of credential in question.
	File string `json:"file"`

	// Url is the URL to call for URL sourced credentials.
	// One field amongst File, URL, Executable, or EnvironmentID should be provided, depending on the kind of credential in question.
	URL string `json:"url"`
	// Headers are the headers to attach to the request for URL sourced credentials.
	Headers map[string]string `json:"headers"`

	// Executable is the configuration object for executable sourced credentials.
	// One field amongst File, URL, Executable, or EnvironmentID should be provided, depending on the kind of credential in question.
	Executable *ExecutableConfig `json:"executable"`

	// EnvironmentID is the EnvironmentID used for AWS sourced credentials. This should start with "AWS".
	// One field amongst File, URL, Executable, or EnvironmentID should be provided, depending on the kind of credential in question.
	EnvironmentID string `json:"environment_id"`
	// RegionURL is the metadata URL to retrieve the region from for EC2 AWS credentials.
	RegionURL string `json:"region_url"`
	// RegionalCredVerificationURL is the AWS regional credential verification URL, will default to
	//  "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15" if not provided."
	RegionalCredVerificationURL string `json:"regional_cred_verification_url"`
	// IMDSv2SessionTokenURL is the URL to retrieve the session token when using IMDSv2 in AWS.
	IMDSv2SessionTokenURL string `json:"imdsv2_session_token_url"`
	// Format is the format type for the subject token. Used for File and URL sourced credentials. Expected values are "text" or "json".
	Format Format `json:"format"`
}

// ExecutableConfig contains information needed for executable sourced credentials.
type ExecutableConfig struct {
	// Command is the the full command to run to retrieve the subject token.
	// This can include arguments. Must be an absolute path for the program. Required.
	Command string `json:"command"`
	// TimeoutMillis is the timeout duration, in milliseconds. Defaults to 30000 milliseconds when not provided. Optional.
	TimeoutMillis *int `json:"timeout_millis"`
	// OutputFile is the absolute path to the output file where the executable will cache the response.
	// If specified the auth libraries will first check this location before running the executable. Optional.
	OutputFile string `json:"output_file"`
}

// SubjectTokenSupplier can be used to supply a subject token to exchange for a GCP access token.
type SubjectTokenSupplier interface {
	// SubjectToken should return a valid subject token or an error.
	// The external account token source does not cache the returned subject token, so caching
	// logic should be implemented in the supplier to prevent multiple requests for the same subject token.
	SubjectToken(ctx context.Context, options SupplierOptions) (string, error)
}

// AWSSecurityCredentialsSupplier can be used to supply AwsSecurityCredentials and an AWS Region to
// exchange for a GCP access token.
type AwsSecurityCredentialsSupplier interface {
	// AwsRegion should return the AWS region or an error.
	AwsRegion(ctx context.Context, options SupplierOptions) (string, error)
	// GetAwsSecurityCredentials should return a valid set of AwsSecurityCredentials or an error.
	// The external account token source does not cache the returned security credentials, so caching
	// logic should be implemented in the supplier to prevent multiple requests for the same security credentials.
	AwsSecurityCredentials(ctx context.Context, options SupplierOptions) (*AwsSecurityCredentials, error)
}

// SupplierOptions contains information about the requested subject token or AWS security credentials from the
// Google external account credential.
type SupplierOptions struct {
	// Audience is the requested audience for the external account credential.
	Audience string
	// Subject token type is the requested subject token type for the external account credential. Expected values include:
	// “urn:ietf:params:oauth:token-type:jwt”
	// “urn:ietf:params:oauth:token-type:id-token”
	// “urn:ietf:params:oauth:token-type:saml2”
	// “urn:ietf:params:aws:token-type:aws4_request”
	SubjectTokenType string
}

// tokenURL returns the default STS token endpoint with the configured universe
// domain.
func (c *Config) tokenURL() string {
	if c.UniverseDomain == "" {
		return strings.Replace(defaultTokenURL, universeDomainPlaceholder, defaultUniverseDomain, 1)
	}
	return strings.Replace(defaultTokenURL, universeDomainPlaceholder, c.UniverseDomain, 1)
}

// parse determines the type of CredentialSource needed.
func (c *Config) parse(ctx context.Context) (baseCredentialSource, error) {
	//set Defaults
	if c.TokenURL == "" {
		c.TokenURL = c.tokenURL()
	}
	supplierOptions := SupplierOptions{Audience: c.Audience, SubjectTokenType: c.SubjectTokenType}

	if c.AwsSecurityCredentialsSupplier != nil {
		awsCredSource := awsCredentialSource{
			awsSecurityCredentialsSupplier: c.AwsSecurityCredentialsSupplier,
			targetResource:                 c.Audience,
			supplierOptions:                supplierOptions,
			ctx:                            ctx,
		}
		return awsCredSource, nil
	} else if c.SubjectTokenSupplier != nil {
		return programmaticRefreshCredentialSource{subjectTokenSupplier: c.SubjectTokenSupplier, supplierOptions: supplierOptions, ctx: ctx}, nil
	} else if len(c.CredentialSource.EnvironmentID) > 3 && c.CredentialSource.EnvironmentID[:3] == "aws" {
		if awsVersion, err := strconv.Atoi(c.CredentialSource.EnvironmentID[3:]); err == nil {
			if awsVersion != 1 {
				return nil, fmt.Errorf("oauth2/google/externalaccount: aws version '%d' is not supported in the current build", awsVersion)
			}

			awsCredSource := awsCredentialSource{
				environmentID:               c.CredentialSource.EnvironmentID,
				regionURL:                   c.CredentialSource.RegionURL,
				regionalCredVerificationURL: c.CredentialSource.RegionalCredVerificationURL,
				credVerificationURL:         c.CredentialSource.URL,
				targetResource:              c.Audience,
				ctx:                         ctx,
			}
			if c.CredentialSource.IMDSv2SessionTokenURL != "" {
				awsCredSource.imdsv2SessionTokenURL = c.CredentialSource.IMDSv2SessionTokenURL
			}

			return awsCredSource, nil
		}
	} else if c.CredentialSource.File != "" {
		return fileCredentialSource{File: c.CredentialSource.File, Format: c.CredentialSource.Format}, nil
	} else if c.CredentialSource.URL != "" {
		return urlCredentialSource{URL: c.CredentialSource.URL, Headers: c.CredentialSource.Headers, Format: c.CredentialSource.Format, ctx: ctx}, nil
	} else if c.CredentialSource.Executable != nil {
		return createExecutableCredential(ctx, c.CredentialSource.Executable, c)
	}
	return nil, fmt.Errorf("oauth2/google/externalaccount: unable to parse credential source")
}

type baseCredentialSource interface {
	credentialSourceType() string
	subjectToken() (string, error)
}

// tokenSource is the source that handles external credentials. It is used to retrieve Tokens.
type tokenSource struct {
	ctx  context.Context
	conf *Config
}

func getMetricsHeaderValue(conf *Config, credSource baseCredentialSource) string {
	return fmt.Sprintf("gl-go/%s auth/%s google-byoid-sdk source/%s sa-impersonation/%t config-lifetime/%t",
		goVersion(),
		"unknown",
		credSource.credentialSourceType(),
		conf.ServiceAccountImpersonationURL != "",
		conf.ServiceAccountImpersonationLifetimeSeconds != 0)
}

// Token allows tokenSource to conform to the oauth2.TokenSource interface.
func (ts tokenSource) Token() (*oauth2.Token, error) {
	conf := ts.conf

	credSource, err := conf.parse(ts.ctx)
	if err != nil {
		return nil, err
	}
	subjectToken, err := credSource.subjectToken()

	if err != nil {
		return nil, err
	}
	stsRequest := stsexchange.TokenExchangeRequest{
		GrantType:          "urn:ietf:params:oauth:grant-type:token-exchange",
		Audience:           conf.Audience,
		Scope:              conf.Scopes,
		RequestedTokenType: "urn:ietf:params:oauth:token-type:access_token",
		SubjectToken:       subjectToken,
		SubjectTokenType:   conf.SubjectTokenType,
	}
	header := make(http.Header)
	header.Add("Content-Type", "application/x-www-form-urlencoded")
	header.Add("x-goog-api-client", getMetricsHeaderValue(conf, credSource))
	clientAuth := stsexchange.ClientAuthentication{
		AuthStyle:    oauth2.AuthStyleInHeader,
		ClientID:     conf.ClientID,
		ClientSecret: conf.ClientSecret,
	}
	var options map[string]interface{}
	// Do not pass workforce_pool_user_project when client authentication is used.
	// The client ID is sufficient for determining the user project.
	if conf.WorkforcePoolUserProject != "" && conf.ClientID == "" {
		options = map[string]interface{}{
			"userProject": conf.WorkforcePoolUserProject,
		}
	}
	stsResp, err := stsexchange.ExchangeToken(ts.ctx, conf.TokenURL, &stsRequest, clientAuth, header, options)
	if err != nil {
		return nil, err
	}

	accessToken := &oauth2.Token{
		AccessToken: stsResp.AccessToken,
		TokenType:   stsResp.TokenType,
	}
	if stsResp.ExpiresIn < 0 {
		return nil, fmt.Errorf("oauth2/google/externalaccount: got invalid expiry from security token service")
	} else if stsResp.ExpiresIn >= 0 {
		accessToken.Expiry = now().Add(time.Duration(stsResp.ExpiresIn) * time.Second)
	}

	if stsResp.RefreshToken != "" {
		accessToken.RefreshToken = stsResp.RefreshToken
	}
	return accessToken, nil
}
