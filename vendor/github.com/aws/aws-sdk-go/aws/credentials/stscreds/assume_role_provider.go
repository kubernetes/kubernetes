// Package stscreds are credential Providers to retrieve STS AWS credentials.
//
// STS provides multiple ways to retrieve credentials which can be used when making
// future AWS service API operation calls.
package stscreds

import (
	"fmt"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/service/sts"
)

// AssumeRoler represents the minimal subset of the STS client API used by this provider.
type AssumeRoler interface {
	AssumeRole(input *sts.AssumeRoleInput) (*sts.AssumeRoleOutput, error)
}

// DefaultDuration is the default amount of time in minutes that the credentials
// will be valid for.
var DefaultDuration = time.Duration(15) * time.Minute

// AssumeRoleProvider retrieves temporary credentials from the STS service, and
// keeps track of their expiration time. This provider must be used explicitly,
// as it is not included in the credentials chain.
type AssumeRoleProvider struct {
	credentials.Expiry

	// STS client to make assume role request with.
	Client AssumeRoler

	// Role to be assumed.
	RoleARN string

	// Session name, if you wish to reuse the credentials elsewhere.
	RoleSessionName string

	// Expiry duration of the STS credentials. Defaults to 15 minutes if not set.
	Duration time.Duration

	// Optional ExternalID to pass along, defaults to nil if not set.
	ExternalID *string

	// ExpiryWindow will allow the credentials to trigger refreshing prior to
	// the credentials actually expiring. This is beneficial so race conditions
	// with expiring credentials do not cause request to fail unexpectedly
	// due to ExpiredTokenException exceptions.
	//
	// So a ExpiryWindow of 10s would cause calls to IsExpired() to return true
	// 10 seconds before the credentials are actually expired.
	//
	// If ExpiryWindow is 0 or less it will be ignored.
	ExpiryWindow time.Duration
}

// NewCredentials returns a pointer to a new Credentials object wrapping the
// AssumeRoleProvider. The credentials will expire every 15 minutes and the
// role will be named after a nanosecond timestamp of this operation.
//
// Takes a Config provider to create the STS client. The ConfigProvider is
// satisfied by the session.Session type.
func NewCredentials(c client.ConfigProvider, roleARN string, options ...func(*AssumeRoleProvider)) *credentials.Credentials {
	p := &AssumeRoleProvider{
		Client:   sts.New(c),
		RoleARN:  roleARN,
		Duration: DefaultDuration,
	}

	for _, option := range options {
		option(p)
	}

	return credentials.NewCredentials(p)
}

// NewCredentialsWithClient returns a pointer to a new Credentials object wrapping the
// AssumeRoleProvider. The credentials will expire every 15 minutes and the
// role will be named after a nanosecond timestamp of this operation.
//
// Takes an AssumeRoler which can be satisfiede by the STS client.
func NewCredentialsWithClient(svc AssumeRoler, roleARN string, options ...func(*AssumeRoleProvider)) *credentials.Credentials {
	p := &AssumeRoleProvider{
		Client:   svc,
		RoleARN:  roleARN,
		Duration: DefaultDuration,
	}

	for _, option := range options {
		option(p)
	}

	return credentials.NewCredentials(p)
}

// Retrieve generates a new set of temporary credentials using STS.
func (p *AssumeRoleProvider) Retrieve() (credentials.Value, error) {

	// Apply defaults where parameters are not set.
	if p.RoleSessionName == "" {
		// Try to work out a role name that will hopefully end up unique.
		p.RoleSessionName = fmt.Sprintf("%d", time.Now().UTC().UnixNano())
	}
	if p.Duration == 0 {
		// Expire as often as AWS permits.
		p.Duration = DefaultDuration
	}

	roleOutput, err := p.Client.AssumeRole(&sts.AssumeRoleInput{
		DurationSeconds: aws.Int64(int64(p.Duration / time.Second)),
		RoleArn:         aws.String(p.RoleARN),
		RoleSessionName: aws.String(p.RoleSessionName),
		ExternalId:      p.ExternalID,
	})

	if err != nil {
		return credentials.Value{}, err
	}

	// We will proactively generate new credentials before they expire.
	p.SetExpiration(*roleOutput.Credentials.Expiration, p.ExpiryWindow)

	return credentials.Value{
		AccessKeyID:     *roleOutput.Credentials.AccessKeyId,
		SecretAccessKey: *roleOutput.Credentials.SecretAccessKey,
		SessionToken:    *roleOutput.Credentials.SessionToken,
	}, nil
}
