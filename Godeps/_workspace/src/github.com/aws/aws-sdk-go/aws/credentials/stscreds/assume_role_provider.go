// Package stscreds are credential Providers to retrieve STS AWS credentials.
//
// STS provides multiple ways to retrieve credentials which can be used when making
// future AWS service API operation calls.
package stscreds

import (
	"fmt"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/service/sts"
)

// AssumeRoler represents the minimal subset of the STS client API used by this provider.
type AssumeRoler interface {
	AssumeRole(input *sts.AssumeRoleInput) (*sts.AssumeRoleOutput, error)
}

// AssumeRoleProvider retrieves temporary credentials from the STS service, and
// keeps track of their expiration time. This provider must be used explicitly,
// as it is not included in the credentials chain.
//
// Example how to configure a service to use this provider:
//
//		config := &aws.Config{
//			Credentials: stscreds.NewCredentials(nil, "arn-of-the-role-to-assume", 10*time.Second),
//		})
//		// Use config for creating your AWS service.
//
// Example how to obtain customised credentials:
//
//		provider := &stscreds.Provider{
//			// Extend the duration to 1 hour.
//			Duration: time.Hour,
//			// Custom role name.
//			RoleSessionName: "custom-session-name",
//		}
//		creds := credentials.NewCredentials(provider)
//
type AssumeRoleProvider struct {
	credentials.Expiry

	// Custom STS client. If not set the default STS client will be used.
	Client AssumeRoler

	// Role to be assumed.
	RoleARN string

	// Session name, if you wish to reuse the credentials elsewhere.
	RoleSessionName string

	// Expiry duration of the STS credentials. Defaults to 15 minutes if not set.
	Duration time.Duration

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
// AssumeRoleProvider.  The credentials will expire every 15 minutes and the
// role will be named after a nanosecond timestamp of this operation.
//
// The sts and roleARN parameters are used for building the "AssumeRole" call.
// Pass nil as sts to use the default client.
//
// Window is the expiry window that will be subtracted from the expiry returned
// by the role credential request. This is done so that the credentials will
// expire sooner than their actual lifespan.
func NewCredentials(client AssumeRoler, roleARN string, window time.Duration) *credentials.Credentials {
	return credentials.NewCredentials(&AssumeRoleProvider{
		Client:       client,
		RoleARN:      roleARN,
		ExpiryWindow: window,
	})
}

// Retrieve generates a new set of temporary credentials using STS.
func (p *AssumeRoleProvider) Retrieve() (credentials.Value, error) {

	// Apply defaults where parameters are not set.
	if p.Client == nil {
		p.Client = sts.New(nil)
	}
	if p.RoleSessionName == "" {
		// Try to work out a role name that will hopefully end up unique.
		p.RoleSessionName = fmt.Sprintf("%d", time.Now().UTC().UnixNano())
	}
	if p.Duration == 0 {
		// Expire as often as AWS permits.
		p.Duration = 15 * time.Minute
	}

	roleOutput, err := p.Client.AssumeRole(&sts.AssumeRoleInput{
		DurationSeconds: aws.Int64(int64(p.Duration / time.Second)),
		RoleArn:         aws.String(p.RoleARN),
		RoleSessionName: aws.String(p.RoleSessionName),
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
