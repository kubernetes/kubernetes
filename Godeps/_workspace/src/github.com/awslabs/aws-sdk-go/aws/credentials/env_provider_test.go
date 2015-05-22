package credentials

import (
	"github.com/stretchr/testify/assert"
	"os"
	"testing"
)

func TestEnvProviderRetrieve(t *testing.T) {
	os.Clearenv()
	os.Setenv("AWS_ACCESS_KEY_ID", "access")
	os.Setenv("AWS_SECRET_ACCESS_KEY", "secret")
	os.Setenv("AWS_SESSION_TOKEN", "token")

	e := EnvProvider{}
	creds, err := e.Retrieve()
	assert.Nil(t, err, "Expect no error", err)

	assert.Equal(t, "access", creds.AccessKeyID, "Expect access key ID to match")
	assert.Equal(t, "secret", creds.SecretAccessKey, "Expect secret access key to match")
	assert.Equal(t, "token", creds.SessionToken, "Expect session token to match")
}

func TestEnvProviderIsExpired(t *testing.T) {
	os.Clearenv()
	os.Setenv("AWS_ACCESS_KEY_ID", "access")
	os.Setenv("AWS_SECRET_ACCESS_KEY", "secret")
	os.Setenv("AWS_SESSION_TOKEN", "token")

	e := EnvProvider{}

	assert.True(t, e.IsExpired(), "Expect creds to be expired before retrieve.")

	_, err := e.Retrieve()
	assert.Nil(t, err, "Expect no error", err)

	assert.False(t, e.IsExpired(), "Expect creds to not be expired after retrieve.")
}

func TestEnvProviderNoAccessKeyID(t *testing.T) {
	os.Clearenv()
	os.Setenv("AWS_SECRET_ACCESS_KEY", "secret")

	e := EnvProvider{}
	creds, err := e.Retrieve()
	assert.Equal(t, ErrAccessKeyIDNotFound, err, "ErrAccessKeyIDNotFound expected, but was %#v error: %#v", creds, err)
}

func TestEnvProviderNoSecretAccessKey(t *testing.T) {
	os.Clearenv()
	os.Setenv("AWS_ACCESS_KEY_ID", "access")

	e := EnvProvider{}
	creds, err := e.Retrieve()
	assert.Equal(t, ErrSecretAccessKeyNotFound, err, "ErrSecretAccessKeyNotFound expected, but was %#v error: %#v", creds, err)
}

func TestEnvProviderAlternateNames(t *testing.T) {
	os.Clearenv()
	os.Setenv("AWS_ACCESS_KEY", "access")
	os.Setenv("AWS_SECRET_KEY", "secret")

	e := EnvProvider{}
	creds, err := e.Retrieve()
	assert.Nil(t, err, "Expect no error")

	assert.Equal(t, "access", creds.AccessKeyID, "Expected access key ID")
	assert.Equal(t, "secret", creds.SecretAccessKey, "Expected secret access key")
	assert.Empty(t, creds.SessionToken, "Expected no token")
}
