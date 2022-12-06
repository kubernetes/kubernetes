// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package externalaccount

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"

	"golang.org/x/oauth2"
)

// now aliases time.Now for testing
var now = func() time.Time {
	return time.Now().UTC()
}

// Config stores the configuration for fetching tokens with external credentials.
type Config struct {
	// Audience is the Secure Token Service (STS) audience which contains the resource name for the workload
	// identity pool or the workforce pool and the provider identifier in that pool.
	Audience string
	// SubjectTokenType is the STS token type based on the Oauth2.0 token exchange spec
	// e.g. `urn:ietf:params:oauth:token-type:jwt`.
	SubjectTokenType string
	// TokenURL is the STS token exchange endpoint.
	TokenURL string
	// TokenInfoURL is the token_info endpoint used to retrieve the account related information (
	// user attributes like account identifier, eg. email, username, uid, etc). This is
	// needed for gCloud session account identification.
	TokenInfoURL string
	// ServiceAccountImpersonationURL is the URL for the service account impersonation request. This is only
	// required for workload identity pools when APIs to be accessed have not integrated with UberMint.
	ServiceAccountImpersonationURL string
	// ClientSecret is currently only required if token_info endpoint also
	// needs to be called with the generated GCP access token. When provided, STS will be
	// called with additional basic authentication using client_id as username and client_secret as password.
	ClientSecret string
	// ClientID is only required in conjunction with ClientSecret, as described above.
	ClientID string
	// CredentialSource contains the necessary information to retrieve the token itself, as well
	// as some environmental information.
	CredentialSource CredentialSource
	// QuotaProjectID is injected by gCloud. If the value is non-empty, the Auth libraries
	// will set the x-goog-user-project which overrides the project associated with the credentials.
	QuotaProjectID string
	// Scopes contains the desired scopes for the returned access token.
	Scopes []string
	// The optional workforce pool user project number when the credential
	// corresponds to a workforce pool and not a workload identity pool.
	// The underlying principal must still have serviceusage.services.use IAM
	// permission to use the project for billing/quota.
	WorkforcePoolUserProject string
}

// Each element consists of a list of patterns.  validateURLs checks for matches
// that include all elements in a given list, in that order.

var (
	validTokenURLPatterns = []*regexp.Regexp{
		// The complicated part in the middle matches any number of characters that
		// aren't period, spaces, or slashes.
		regexp.MustCompile(`(?i)^[^\.\s\/\\]+\.sts\.googleapis\.com$`),
		regexp.MustCompile(`(?i)^sts\.googleapis\.com$`),
		regexp.MustCompile(`(?i)^sts\.[^\.\s\/\\]+\.googleapis\.com$`),
		regexp.MustCompile(`(?i)^[^\.\s\/\\]+-sts\.googleapis\.com$`),
	}
	validImpersonateURLPatterns = []*regexp.Regexp{
		regexp.MustCompile(`^[^\.\s\/\\]+\.iamcredentials\.googleapis\.com$`),
		regexp.MustCompile(`^iamcredentials\.googleapis\.com$`),
		regexp.MustCompile(`^iamcredentials\.[^\.\s\/\\]+\.googleapis\.com$`),
		regexp.MustCompile(`^[^\.\s\/\\]+-iamcredentials\.googleapis\.com$`),
	}
	validWorkforceAudiencePattern *regexp.Regexp = regexp.MustCompile(`//iam\.googleapis\.com/locations/[^/]+/workforcePools/`)
)

func validateURL(input string, patterns []*regexp.Regexp, scheme string) bool {
	parsed, err := url.Parse(input)
	if err != nil {
		return false
	}
	if !strings.EqualFold(parsed.Scheme, scheme) {
		return false
	}
	toTest := parsed.Host

	for _, pattern := range patterns {
		if pattern.MatchString(toTest) {
			return true
		}
	}
	return false
}

func validateWorkforceAudience(input string) bool {
	return validWorkforceAudiencePattern.MatchString(input)
}

// TokenSource Returns an external account TokenSource struct. This is to be called by package google to construct a google.Credentials.
func (c *Config) TokenSource(ctx context.Context) (oauth2.TokenSource, error) {
	return c.tokenSource(ctx, validTokenURLPatterns, validImpersonateURLPatterns, "https")
}

// tokenSource is a private function that's directly called by some of the tests,
// because the unit test URLs are mocked, and would otherwise fail the
// validity check.
func (c *Config) tokenSource(ctx context.Context, tokenURLValidPats []*regexp.Regexp, impersonateURLValidPats []*regexp.Regexp, scheme string) (oauth2.TokenSource, error) {
	valid := validateURL(c.TokenURL, tokenURLValidPats, scheme)
	if !valid {
		return nil, fmt.Errorf("oauth2/google: invalid TokenURL provided while constructing tokenSource")
	}

	if c.ServiceAccountImpersonationURL != "" {
		valid := validateURL(c.ServiceAccountImpersonationURL, impersonateURLValidPats, scheme)
		if !valid {
			return nil, fmt.Errorf("oauth2/google: invalid ServiceAccountImpersonationURL provided while constructing tokenSource")
		}
	}

	if c.WorkforcePoolUserProject != "" {
		valid := validateWorkforceAudience(c.Audience)
		if !valid {
			return nil, fmt.Errorf("oauth2/google: workforce_pool_user_project should not be set for non-workforce pool credentials")
		}
	}

	ts := tokenSource{
		ctx:  ctx,
		conf: c,
	}
	if c.ServiceAccountImpersonationURL == "" {
		return oauth2.ReuseTokenSource(nil, ts), nil
	}
	scopes := c.Scopes
	ts.conf.Scopes = []string{"https://www.googleapis.com/auth/cloud-platform"}
	imp := ImpersonateTokenSource{
		Ctx:    ctx,
		URL:    c.ServiceAccountImpersonationURL,
		Scopes: scopes,
		Ts:     oauth2.ReuseTokenSource(nil, ts),
	}
	return oauth2.ReuseTokenSource(nil, imp), nil
}

// Subject token file types.
const (
	fileTypeText = "text"
	fileTypeJSON = "json"
)

type format struct {
	// Type is either "text" or "json". When not provided "text" type is assumed.
	Type string `json:"type"`
	// SubjectTokenFieldName is only required for JSON format. This would be "access_token" for azure.
	SubjectTokenFieldName string `json:"subject_token_field_name"`
}

// CredentialSource stores the information necessary to retrieve the credentials for the STS exchange.
// Either the File or the URL field should be filled, depending on the kind of credential in question.
// The EnvironmentID should start with AWS if being used for an AWS credential.
type CredentialSource struct {
	File string `json:"file"`

	URL     string            `json:"url"`
	Headers map[string]string `json:"headers"`

	EnvironmentID               string `json:"environment_id"`
	RegionURL                   string `json:"region_url"`
	RegionalCredVerificationURL string `json:"regional_cred_verification_url"`
	CredVerificationURL         string `json:"cred_verification_url"`
	Format                      format `json:"format"`
}

// parse determines the type of CredentialSource needed
func (c *Config) parse(ctx context.Context) (baseCredentialSource, error) {
	if len(c.CredentialSource.EnvironmentID) > 3 && c.CredentialSource.EnvironmentID[:3] == "aws" {
		if awsVersion, err := strconv.Atoi(c.CredentialSource.EnvironmentID[3:]); err == nil {
			if awsVersion != 1 {
				return nil, fmt.Errorf("oauth2/google: aws version '%d' is not supported in the current build", awsVersion)
			}
			return awsCredentialSource{
				EnvironmentID:               c.CredentialSource.EnvironmentID,
				RegionURL:                   c.CredentialSource.RegionURL,
				RegionalCredVerificationURL: c.CredentialSource.RegionalCredVerificationURL,
				CredVerificationURL:         c.CredentialSource.URL,
				TargetResource:              c.Audience,
				ctx:                         ctx,
			}, nil
		}
	} else if c.CredentialSource.File != "" {
		return fileCredentialSource{File: c.CredentialSource.File, Format: c.CredentialSource.Format}, nil
	} else if c.CredentialSource.URL != "" {
		return urlCredentialSource{URL: c.CredentialSource.URL, Headers: c.CredentialSource.Headers, Format: c.CredentialSource.Format, ctx: ctx}, nil
	}
	return nil, fmt.Errorf("oauth2/google: unable to parse credential source")
}

type baseCredentialSource interface {
	subjectToken() (string, error)
}

// tokenSource is the source that handles external credentials. It is used to retrieve Tokens.
type tokenSource struct {
	ctx  context.Context
	conf *Config
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
	stsRequest := stsTokenExchangeRequest{
		GrantType:          "urn:ietf:params:oauth:grant-type:token-exchange",
		Audience:           conf.Audience,
		Scope:              conf.Scopes,
		RequestedTokenType: "urn:ietf:params:oauth:token-type:access_token",
		SubjectToken:       subjectToken,
		SubjectTokenType:   conf.SubjectTokenType,
	}
	header := make(http.Header)
	header.Add("Content-Type", "application/x-www-form-urlencoded")
	clientAuth := clientAuthentication{
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
	stsResp, err := exchangeToken(ts.ctx, conf.TokenURL, &stsRequest, clientAuth, header, options)
	if err != nil {
		return nil, err
	}

	accessToken := &oauth2.Token{
		AccessToken: stsResp.AccessToken,
		TokenType:   stsResp.TokenType,
	}
	if stsResp.ExpiresIn < 0 {
		return nil, fmt.Errorf("oauth2/google: got invalid expiry from security token service")
	} else if stsResp.ExpiresIn >= 0 {
		accessToken.Expiry = now().Add(time.Duration(stsResp.ExpiresIn) * time.Second)
	}

	if stsResp.RefreshToken != "" {
		accessToken.RefreshToken = stsResp.RefreshToken
	}
	return accessToken, nil
}
