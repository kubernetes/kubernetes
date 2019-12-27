/*
Package stscreds are credential Providers to retrieve STS AWS credentials.

STS provides multiple ways to retrieve credentials which can be used when making
future AWS service API operation calls.

The SDK will ensure that per instance of credentials.Credentials all requests
to refresh the credentials will be synchronized. But, the SDK is unable to
ensure synchronous usage of the AssumeRoleProvider if the value is shared
between multiple Credentials, Sessions or service clients.

Assume Role

To assume an IAM role using STS with the SDK you can create a new Credentials
with the SDKs's stscreds package.

	// Initial credentials loaded from SDK's default credential chain. Such as
	// the environment, shared credentials (~/.aws/credentials), or EC2 Instance
	// Role. These credentials will be used to to make the STS Assume Role API.
	sess := session.Must(session.NewSession())

	// Create the credentials from AssumeRoleProvider to assume the role
	// referenced by the "myRoleARN" ARN.
	creds := stscreds.NewCredentials(sess, "myRoleArn")

	// Create service client value configured for credentials
	// from assumed role.
	svc := s3.New(sess, &aws.Config{Credentials: creds})

Assume Role with static MFA Token

To assume an IAM role with a MFA token you can either specify a MFA token code
directly or provide a function to prompt the user each time the credentials
need to refresh the role's credentials. Specifying the TokenCode should be used
for short lived operations that will not need to be refreshed, and when you do
not want to have direct control over the user provides their MFA token.

With TokenCode the AssumeRoleProvider will be not be able to refresh the role's
credentials.

	// Create the credentials from AssumeRoleProvider to assume the role
	// referenced by the "myRoleARN" ARN using the MFA token code provided.
	creds := stscreds.NewCredentials(sess, "myRoleArn", func(p *stscreds.AssumeRoleProvider) {
		p.SerialNumber = aws.String("myTokenSerialNumber")
		p.TokenCode = aws.String("00000000")
	})

	// Create service client value configured for credentials
	// from assumed role.
	svc := s3.New(sess, &aws.Config{Credentials: creds})

Assume Role with MFA Token Provider

To assume an IAM role with MFA for longer running tasks where the credentials
may need to be refreshed setting the TokenProvider field of AssumeRoleProvider
will allow the credential provider to prompt for new MFA token code when the
role's credentials need to be refreshed.

The StdinTokenProvider function is available to prompt on stdin to retrieve
the MFA token code from the user. You can also implement custom prompts by
satisfing the TokenProvider function signature.

Using StdinTokenProvider with multiple AssumeRoleProviders, or Credentials will
have undesirable results as the StdinTokenProvider will not be synchronized. A
single Credentials with an AssumeRoleProvider can be shared safely.

	// Create the credentials from AssumeRoleProvider to assume the role
	// referenced by the "myRoleARN" ARN. Prompting for MFA token from stdin.
	creds := stscreds.NewCredentials(sess, "myRoleArn", func(p *stscreds.AssumeRoleProvider) {
		p.SerialNumber = aws.String("myTokenSerialNumber")
		p.TokenProvider = stscreds.StdinTokenProvider
	})

	// Create service client value configured for credentials
	// from assumed role.
	svc := s3.New(sess, &aws.Config{Credentials: creds})

*/
package stscreds

import (
	"fmt"
	"os"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/internal/sdkrand"
	"github.com/aws/aws-sdk-go/service/sts"
)

// StdinTokenProvider will prompt on stderr and read from stdin for a string value.
// An error is returned if reading from stdin fails.
//
// Use this function go read MFA tokens from stdin. The function makes no attempt
// to make atomic prompts from stdin across multiple gorouties.
//
// Using StdinTokenProvider with multiple AssumeRoleProviders, or Credentials will
// have undesirable results as the StdinTokenProvider will not be synchronized. A
// single Credentials with an AssumeRoleProvider can be shared safely
//
// Will wait forever until something is provided on the stdin.
func StdinTokenProvider() (string, error) {
	var v string
	fmt.Fprintf(os.Stderr, "Assume Role MFA token code: ")
	_, err := fmt.Scanln(&v)

	return v, err
}

// ProviderName provides a name of AssumeRole provider
const ProviderName = "AssumeRoleProvider"

// AssumeRoler represents the minimal subset of the STS client API used by this provider.
type AssumeRoler interface {
	AssumeRole(input *sts.AssumeRoleInput) (*sts.AssumeRoleOutput, error)
}

// DefaultDuration is the default amount of time in minutes that the credentials
// will be valid for.
var DefaultDuration = time.Duration(15) * time.Minute

// AssumeRoleProvider retrieves temporary credentials from the STS service, and
// keeps track of their expiration time.
//
// This credential provider will be used by the SDKs default credential change
// when shared configuration is enabled, and the shared config or shared credentials
// file configure assume role. See Session docs for how to do this.
//
// AssumeRoleProvider does not provide any synchronization and it is not safe
// to share this value across multiple Credentials, Sessions, or service clients
// without also sharing the same Credentials instance.
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

	// The policy plain text must be 2048 bytes or shorter. However, an internal
	// conversion compresses it into a packed binary format with a separate limit.
	// The PackedPolicySize response element indicates by percentage how close to
	// the upper size limit the policy is, with 100% equaling the maximum allowed
	// size.
	Policy *string

	// The identification number of the MFA device that is associated with the user
	// who is making the AssumeRole call. Specify this value if the trust policy
	// of the role being assumed includes a condition that requires MFA authentication.
	// The value is either the serial number for a hardware device (such as GAHT12345678)
	// or an Amazon Resource Name (ARN) for a virtual device (such as arn:aws:iam::123456789012:mfa/user).
	SerialNumber *string

	// The value provided by the MFA device, if the trust policy of the role being
	// assumed requires MFA (that is, if the policy includes a condition that tests
	// for MFA). If the role being assumed requires MFA and if the TokenCode value
	// is missing or expired, the AssumeRole call returns an "access denied" error.
	//
	// If SerialNumber is set and neither TokenCode nor TokenProvider are also
	// set an error will be returned.
	TokenCode *string

	// Async method of providing MFA token code for assuming an IAM role with MFA.
	// The value returned by the function will be used as the TokenCode in the Retrieve
	// call. See StdinTokenProvider for a provider that prompts and reads from stdin.
	//
	// This token provider will be called when ever the assumed role's
	// credentials need to be refreshed when SerialNumber is also set and
	// TokenCode is not set.
	//
	// If both TokenCode and TokenProvider is set, TokenProvider will be used and
	// TokenCode is ignored.
	TokenProvider func() (string, error)

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

	// MaxJitterFrac reduces the effective Duration of each credential requested
	// by a random percentage between 0 and MaxJitterFraction. MaxJitterFrac must
	// have a value between 0 and 1. Any other value may lead to expected behavior.
	// With a MaxJitterFrac value of 0, default) will no jitter will be used.
	//
	// For example, with a Duration of 30m and a MaxJitterFrac of 0.1, the
	// AssumeRole call will be made with an arbitrary Duration between 27m and
	// 30m.
	//
	// MaxJitterFrac should not be negative.
	MaxJitterFrac float64
}

// NewCredentials returns a pointer to a new Credentials object wrapping the
// AssumeRoleProvider. The credentials will expire every 15 minutes and the
// role will be named after a nanosecond timestamp of this operation.
//
// Takes a Config provider to create the STS client. The ConfigProvider is
// satisfied by the session.Session type.
//
// It is safe to share the returned Credentials with multiple Sessions and
// service clients. All access to the credentials and refreshing them
// will be synchronized.
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
// Takes an AssumeRoler which can be satisfied by the STS client.
//
// It is safe to share the returned Credentials with multiple Sessions and
// service clients. All access to the credentials and refreshing them
// will be synchronized.
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
	jitter := time.Duration(sdkrand.SeededRand.Float64() * p.MaxJitterFrac * float64(p.Duration))
	input := &sts.AssumeRoleInput{
		DurationSeconds: aws.Int64(int64((p.Duration - jitter) / time.Second)),
		RoleArn:         aws.String(p.RoleARN),
		RoleSessionName: aws.String(p.RoleSessionName),
		ExternalId:      p.ExternalID,
	}
	if p.Policy != nil {
		input.Policy = p.Policy
	}
	if p.SerialNumber != nil {
		if p.TokenCode != nil {
			input.SerialNumber = p.SerialNumber
			input.TokenCode = p.TokenCode
		} else if p.TokenProvider != nil {
			input.SerialNumber = p.SerialNumber
			code, err := p.TokenProvider()
			if err != nil {
				return credentials.Value{ProviderName: ProviderName}, err
			}
			input.TokenCode = aws.String(code)
		} else {
			return credentials.Value{ProviderName: ProviderName},
				awserr.New("AssumeRoleTokenNotAvailable",
					"assume role with MFA enabled, but neither TokenCode nor TokenProvider are set", nil)
		}
	}

	roleOutput, err := p.Client.AssumeRole(input)
	if err != nil {
		return credentials.Value{ProviderName: ProviderName}, err
	}

	// We will proactively generate new credentials before they expire.
	p.SetExpiration(*roleOutput.Credentials.Expiration, p.ExpiryWindow)

	return credentials.Value{
		AccessKeyID:     *roleOutput.Credentials.AccessKeyId,
		SecretAccessKey: *roleOutput.Credentials.SecretAccessKey,
		SessionToken:    *roleOutput.Credentials.SessionToken,
		ProviderName:    ProviderName,
	}, nil
}
