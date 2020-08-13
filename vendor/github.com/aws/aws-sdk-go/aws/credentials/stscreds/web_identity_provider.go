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

// WebIdentityRoleProvider is used to retrieve credentials using
// an OIDC token.
type WebIdentityRoleProvider struct {
	credentials.Expiry

	client       stsiface.STSAPI
	ExpiryWindow time.Duration

	tokenFilePath   string
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
	return &WebIdentityRoleProvider{
		client:          svc,
		tokenFilePath:   path,
		roleARN:         roleARN,
		roleSessionName: roleSessionName,
	}
}

// Retrieve will attempt to assume a role from a token which is located at
// 'WebIdentityTokenFilePath' specified destination and if that is empty an
// error will be returned.
func (p *WebIdentityRoleProvider) Retrieve() (credentials.Value, error) {
	b, err := ioutil.ReadFile(p.tokenFilePath)
	if err != nil {
		errMsg := fmt.Sprintf("unable to read file at %s", p.tokenFilePath)
		return credentials.Value{}, awserr.New(ErrCodeWebIdentity, errMsg, err)
	}

	sessionName := p.roleSessionName
	if len(sessionName) == 0 {
		// session name is used to uniquely identify a session. This simply
		// uses unix time in nanoseconds to uniquely identify sessions.
		sessionName = strconv.FormatInt(now().UnixNano(), 10)
	}
	req, resp := p.client.AssumeRoleWithWebIdentityRequest(&sts.AssumeRoleWithWebIdentityInput{
		RoleArn:          &p.roleARN,
		RoleSessionName:  &sessionName,
		WebIdentityToken: aws.String(string(b)),
	})
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
