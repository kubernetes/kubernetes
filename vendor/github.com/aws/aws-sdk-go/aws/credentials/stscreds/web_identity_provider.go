package stscreds

import (
	"fmt"
	"io/ioutil"
	"strconv"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/service/sts"
	"github.com/aws/aws-sdk-go/service/sts/stsiface"
)

const (
	// ErrCodeWebIdentity will be used as an error code when constructing
	// a new error to be returned during session creation or retrieval.
	ErrCodeWebIdentity = "WebIdentityErr"

	// WebIdentityProviderName is the web identity provider name
	WebIdentityProviderName = "WebIdentityCredentials"
)

// now is used to return a time.Time object representing
// the current time. This can be used to easily test and
// compare test values.
var now = time.Now

// TokenFetcher shuold return WebIdentity token bytes or an error
type TokenFetcher interface {
	FetchToken(credentials.Context) ([]byte, error)
}

// FetchTokenPath is a path to a WebIdentity token file
type FetchTokenPath string

// FetchToken returns a token by reading from the filesystem
func (f FetchTokenPath) FetchToken(ctx credentials.Context) ([]byte, error) {
	data, err := ioutil.ReadFile(string(f))
	if err != nil {
		errMsg := fmt.Sprintf("unable to read file at %s", f)
		return nil, awserr.New(ErrCodeWebIdentity, errMsg, err)
	}
	return data, nil
}

// WebIdentityRoleProvider is used to retrieve credentials using
// an OIDC token.
type WebIdentityRoleProvider struct {
	credentials.Expiry
	PolicyArns []*sts.PolicyDescriptorType

	// Duration the STS credentials will be valid for. Truncated to seconds.
	// If unset, the assumed role will use AssumeRoleWithWebIdentity's default
	// expiry duration. See
	// https://docs.aws.amazon.com/sdk-for-go/api/service/sts/#STS.AssumeRoleWithWebIdentity
	// for more information.
	Duration time.Duration

	// The amount of time the credentials will be refreshed before they expire.
	// This is useful refresh credentials before they expire to reduce risk of
	// using credentials as they expire. If unset, will default to no expiry
	// window.
	ExpiryWindow time.Duration

	client stsiface.STSAPI

	tokenFetcher    TokenFetcher
	roleARN         string
	roleSessionName string
}

// NewWebIdentityCredentials will return a new set of credentials with a given
// configuration, role arn, and token file path.
func NewWebIdentityCredentials(c client.ConfigProvider, roleARN, roleSessionName, path string) *credentials.Credentials {
	svc := sts.New(c)
	p := NewWebIdentityRoleProvider(svc, roleARN, roleSessionName, path)
	return credentials.NewCredentials(p)
}

// NewWebIdentityRoleProvider will return a new WebIdentityRoleProvider with the
// provided stsiface.STSAPI
func NewWebIdentityRoleProvider(svc stsiface.STSAPI, roleARN, roleSessionName, path string) *WebIdentityRoleProvider {
	return NewWebIdentityRoleProviderWithToken(svc, roleARN, roleSessionName, FetchTokenPath(path))
}

// NewWebIdentityRoleProviderWithToken will return a new WebIdentityRoleProvider with the
// provided stsiface.STSAPI and a TokenFetcher
func NewWebIdentityRoleProviderWithToken(svc stsiface.STSAPI, roleARN, roleSessionName string, tokenFetcher TokenFetcher) *WebIdentityRoleProvider {
	return &WebIdentityRoleProvider{
		client:          svc,
		tokenFetcher:    tokenFetcher,
		roleARN:         roleARN,
		roleSessionName: roleSessionName,
	}
}

// Retrieve will attempt to assume a role from a token which is located at
// 'WebIdentityTokenFilePath' specified destination and if that is empty an
// error will be returned.
func (p *WebIdentityRoleProvider) Retrieve() (credentials.Value, error) {
	return p.RetrieveWithContext(aws.BackgroundContext())
}

// RetrieveWithContext will attempt to assume a role from a token which is located at
// 'WebIdentityTokenFilePath' specified destination and if that is empty an
// error will be returned.
func (p *WebIdentityRoleProvider) RetrieveWithContext(ctx credentials.Context) (credentials.Value, error) {
	b, err := p.tokenFetcher.FetchToken(ctx)
	if err != nil {
		return credentials.Value{}, awserr.New(ErrCodeWebIdentity, "failed fetching WebIdentity token: ", err)
	}

	sessionName := p.roleSessionName
	if len(sessionName) == 0 {
		// session name is used to uniquely identify a session. This simply
		// uses unix time in nanoseconds to uniquely identify sessions.
		sessionName = strconv.FormatInt(now().UnixNano(), 10)
	}

	var duration *int64
	if p.Duration != 0 {
		duration = aws.Int64(int64(p.Duration / time.Second))
	}

	req, resp := p.client.AssumeRoleWithWebIdentityRequest(&sts.AssumeRoleWithWebIdentityInput{
		PolicyArns:       p.PolicyArns,
		RoleArn:          &p.roleARN,
		RoleSessionName:  &sessionName,
		WebIdentityToken: aws.String(string(b)),
		DurationSeconds:  duration,
	})

	req.SetContext(ctx)

	// InvalidIdentityToken error is a temporary error that can occur
	// when assuming an Role with a JWT web identity token.
	req.RetryErrorCodes = append(req.RetryErrorCodes, sts.ErrCodeInvalidIdentityTokenException)
	if err := req.Send(); err != nil {
		return credentials.Value{}, awserr.New(ErrCodeWebIdentity, "failed to retrieve credentials", err)
	}

	p.SetExpiration(aws.TimeValue(resp.Credentials.Expiration), p.ExpiryWindow)

	value := credentials.Value{
		AccessKeyID:     aws.StringValue(resp.Credentials.AccessKeyId),
		SecretAccessKey: aws.StringValue(resp.Credentials.SecretAccessKey),
		SessionToken:    aws.StringValue(resp.Credentials.SessionToken),
		ProviderName:    WebIdentityProviderName,
	}
	return value, nil
}
