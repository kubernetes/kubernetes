package credentials

import (
	"testing"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/stretchr/testify/assert"
)

type stubProvider struct {
	creds   Value
	expired bool
	err     error
}

func (s *stubProvider) Retrieve() (Value, error) {
	s.expired = false
	s.creds.ProviderName = "stubProvider"
	return s.creds, s.err
}
func (s *stubProvider) IsExpired() bool {
	return s.expired
}

func TestCredentialsGet(t *testing.T) {
	c := NewCredentials(&stubProvider{
		creds: Value{
			AccessKeyID:     "AKID",
			SecretAccessKey: "SECRET",
			SessionToken:    "",
		},
		expired: true,
	})

	creds, err := c.Get()
	assert.Nil(t, err, "Expected no error")
	assert.Equal(t, "AKID", creds.AccessKeyID, "Expect access key ID to match")
	assert.Equal(t, "SECRET", creds.SecretAccessKey, "Expect secret access key to match")
	assert.Empty(t, creds.SessionToken, "Expect session token to be empty")
}

func TestCredentialsGetWithError(t *testing.T) {
	c := NewCredentials(&stubProvider{err: awserr.New("provider error", "", nil), expired: true})

	_, err := c.Get()
	assert.Equal(t, "provider error", err.(awserr.Error).Code(), "Expected provider error")
}

func TestCredentialsExpire(t *testing.T) {
	stub := &stubProvider{}
	c := NewCredentials(stub)

	stub.expired = false
	assert.True(t, c.IsExpired(), "Expected to start out expired")
	c.Expire()
	assert.True(t, c.IsExpired(), "Expected to be expired")

	c.forceRefresh = false
	assert.False(t, c.IsExpired(), "Expected not to be expired")

	stub.expired = true
	assert.True(t, c.IsExpired(), "Expected to be expired")
}

func TestCredentialsGetWithProviderName(t *testing.T) {
	stub := &stubProvider{}

	c := NewCredentials(stub)

	creds, err := c.Get()
	assert.Nil(t, err, "Expected no error")
	assert.Equal(t, creds.ProviderName, "stubProvider", "Expected provider name to match")
}
