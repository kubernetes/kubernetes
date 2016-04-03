package stscreds

import (
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/sts"
	"github.com/stretchr/testify/assert"
)

type stubSTS struct {
}

func (s *stubSTS) AssumeRole(input *sts.AssumeRoleInput) (*sts.AssumeRoleOutput, error) {
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
