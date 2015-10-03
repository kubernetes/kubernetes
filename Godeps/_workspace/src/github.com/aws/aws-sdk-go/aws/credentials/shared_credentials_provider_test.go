package credentials

import (
	"github.com/stretchr/testify/assert"
	"os"
	"testing"
)

func TestSharedCredentialsProvider(t *testing.T) {
	os.Clearenv()

	p := SharedCredentialsProvider{Filename: "example.ini", Profile: ""}
	creds, err := p.Retrieve()
	assert.Nil(t, err, "Expect no error")

	assert.Equal(t, "accessKey", creds.AccessKeyID, "Expect access key ID to match")
	assert.Equal(t, "secret", creds.SecretAccessKey, "Expect secret access key to match")
	assert.Equal(t, "token", creds.SessionToken, "Expect session token to match")
}

func TestSharedCredentialsProviderIsExpired(t *testing.T) {
	os.Clearenv()

	p := SharedCredentialsProvider{Filename: "example.ini", Profile: ""}

	assert.True(t, p.IsExpired(), "Expect creds to be expired before retrieve")

	_, err := p.Retrieve()
	assert.Nil(t, err, "Expect no error")

	assert.False(t, p.IsExpired(), "Expect creds to not be expired after retrieve")
}

func TestSharedCredentialsProviderWithAWS_SHARED_CREDENTIALS_FILE(t *testing.T) {
	os.Clearenv()
	os.Setenv("AWS_SHARED_CREDENTIALS_FILE", "example.ini")
	p := SharedCredentialsProvider{}
	creds, err := p.Retrieve()

	assert.Nil(t, err, "Expect no error")

	assert.Equal(t, "accessKey", creds.AccessKeyID, "Expect access key ID to match")
	assert.Equal(t, "secret", creds.SecretAccessKey, "Expect secret access key to match")
	assert.Equal(t, "token", creds.SessionToken, "Expect session token to match")
}

func TestSharedCredentialsProviderWithAWS_PROFILE(t *testing.T) {
	os.Clearenv()
	os.Setenv("AWS_PROFILE", "no_token")

	p := SharedCredentialsProvider{Filename: "example.ini", Profile: ""}
	creds, err := p.Retrieve()
	assert.Nil(t, err, "Expect no error")

	assert.Equal(t, "accessKey", creds.AccessKeyID, "Expect access key ID to match")
	assert.Equal(t, "secret", creds.SecretAccessKey, "Expect secret access key to match")
	assert.Empty(t, creds.SessionToken, "Expect no token")
}

func TestSharedCredentialsProviderWithoutTokenFromProfile(t *testing.T) {
	os.Clearenv()

	p := SharedCredentialsProvider{Filename: "example.ini", Profile: "no_token"}
	creds, err := p.Retrieve()
	assert.Nil(t, err, "Expect no error")

	assert.Equal(t, "accessKey", creds.AccessKeyID, "Expect access key ID to match")
	assert.Equal(t, "secret", creds.SecretAccessKey, "Expect secret access key to match")
	assert.Empty(t, creds.SessionToken, "Expect no token")
}

func BenchmarkSharedCredentialsProvider(b *testing.B) {
	os.Clearenv()

	p := SharedCredentialsProvider{Filename: "example.ini", Profile: ""}
	_, err := p.Retrieve()
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := p.Retrieve()
		if err != nil {
			b.Fatal(err)
		}
	}
}
