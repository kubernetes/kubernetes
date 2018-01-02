package stscreds

import (
	"fmt"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/sts"
	"github.com/stretchr/testify/assert"
)

type stubSTS struct {
	TestInput func(*sts.AssumeRoleInput)
}

func (s *stubSTS) AssumeRole(input *sts.AssumeRoleInput) (*sts.AssumeRoleOutput, error) {
	if s.TestInput != nil {
		s.TestInput(input)
	}
	expiry := time.Now().Add(60 * time.Minute)
	return &sts.AssumeRoleOutput{
		Credentials: &sts.Credentials{
			// Just reflect the role arn to the provider.
			AccessKeyId:     input.RoleArn,
			SecretAccessKey: aws.String("assumedSecretAccessKey"),
			SessionToken:    aws.String("assumedSessionToken"),
			Expiration:      &expiry,
		},
	}, nil
}

func TestAssumeRoleProvider(t *testing.T) {
	stub := &stubSTS{}
	p := &AssumeRoleProvider{
		Client:  stub,
		RoleARN: "roleARN",
	}

	creds, err := p.Retrieve()
	assert.Nil(t, err, "Expect no error")

	assert.Equal(t, "roleARN", creds.AccessKeyID, "Expect access key ID to be reflected role ARN")
	assert.Equal(t, "assumedSecretAccessKey", creds.SecretAccessKey, "Expect secret access key to match")
	assert.Equal(t, "assumedSessionToken", creds.SessionToken, "Expect session token to match")
}

func TestAssumeRoleProvider_WithTokenCode(t *testing.T) {
	stub := &stubSTS{
		TestInput: func(in *sts.AssumeRoleInput) {
			assert.Equal(t, "0123456789", *in.SerialNumber)
			assert.Equal(t, "code", *in.TokenCode)
		},
	}
	p := &AssumeRoleProvider{
		Client:       stub,
		RoleARN:      "roleARN",
		SerialNumber: aws.String("0123456789"),
		TokenCode:    aws.String("code"),
	}

	creds, err := p.Retrieve()
	assert.Nil(t, err, "Expect no error")

	assert.Equal(t, "roleARN", creds.AccessKeyID, "Expect access key ID to be reflected role ARN")
	assert.Equal(t, "assumedSecretAccessKey", creds.SecretAccessKey, "Expect secret access key to match")
	assert.Equal(t, "assumedSessionToken", creds.SessionToken, "Expect session token to match")
}

func TestAssumeRoleProvider_WithTokenProvider(t *testing.T) {
	stub := &stubSTS{
		TestInput: func(in *sts.AssumeRoleInput) {
			assert.Equal(t, "0123456789", *in.SerialNumber)
			assert.Equal(t, "code", *in.TokenCode)
		},
	}
	p := &AssumeRoleProvider{
		Client:       stub,
		RoleARN:      "roleARN",
		SerialNumber: aws.String("0123456789"),
		TokenProvider: func() (string, error) {
			return "code", nil
		},
	}

	creds, err := p.Retrieve()
	assert.Nil(t, err, "Expect no error")

	assert.Equal(t, "roleARN", creds.AccessKeyID, "Expect access key ID to be reflected role ARN")
	assert.Equal(t, "assumedSecretAccessKey", creds.SecretAccessKey, "Expect secret access key to match")
	assert.Equal(t, "assumedSessionToken", creds.SessionToken, "Expect session token to match")
}

func TestAssumeRoleProvider_WithTokenProviderError(t *testing.T) {
	stub := &stubSTS{
		TestInput: func(in *sts.AssumeRoleInput) {
			assert.Fail(t, "API request should not of been called")
		},
	}
	p := &AssumeRoleProvider{
		Client:       stub,
		RoleARN:      "roleARN",
		SerialNumber: aws.String("0123456789"),
		TokenProvider: func() (string, error) {
			return "", fmt.Errorf("error occurred")
		},
	}

	creds, err := p.Retrieve()
	assert.Error(t, err)

	assert.Empty(t, creds.AccessKeyID)
	assert.Empty(t, creds.SecretAccessKey)
	assert.Empty(t, creds.SessionToken)
}

func TestAssumeRoleProvider_MFAWithNoToken(t *testing.T) {
	stub := &stubSTS{
		TestInput: func(in *sts.AssumeRoleInput) {
			assert.Fail(t, "API request should not of been called")
		},
	}
	p := &AssumeRoleProvider{
		Client:       stub,
		RoleARN:      "roleARN",
		SerialNumber: aws.String("0123456789"),
	}

	creds, err := p.Retrieve()
	assert.Error(t, err)

	assert.Empty(t, creds.AccessKeyID)
	assert.Empty(t, creds.SecretAccessKey)
	assert.Empty(t, creds.SessionToken)
}

func BenchmarkAssumeRoleProvider(b *testing.B) {
	stub := &stubSTS{}
	p := &AssumeRoleProvider{
		Client:  stub,
		RoleARN: "roleARN",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := p.Retrieve(); err != nil {
			b.Fatal(err)
		}
	}
}
