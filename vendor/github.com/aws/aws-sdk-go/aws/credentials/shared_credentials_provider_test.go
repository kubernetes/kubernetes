package credentials

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/aws/aws-sdk-go/internal/shareddefaults"
	"github.com/stretchr/testify/assert"
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

func TestSharedCredentialsProviderWithAWS_SHARED_CREDENTIALS_FILEAbsPath(t *testing.T) {
	os.Clearenv()
	wd, err := os.Getwd()
	assert.NoError(t, err)
	os.Setenv("AWS_SHARED_CREDENTIALS_FILE", filepath.Join(wd, "example.ini"))
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

func TestSharedCredentialsProviderColonInCredFile(t *testing.T) {
	os.Clearenv()

	p := SharedCredentialsProvider{Filename: "example.ini", Profile: "with_colon"}
	creds, err := p.Retrieve()
	assert.Nil(t, err, "Expect no error")

	assert.Equal(t, "accessKey", creds.AccessKeyID, "Expect access key ID to match")
	assert.Equal(t, "secret", creds.SecretAccessKey, "Expect secret access key to match")
	assert.Empty(t, creds.SessionToken, "Expect no token")
}

func TestSharedCredentialsProvider_DefaultFilename(t *testing.T) {
	os.Clearenv()
	os.Setenv("USERPROFILE", "profile_dir")
	os.Setenv("HOME", "home_dir")

	// default filename and profile
	p := SharedCredentialsProvider{}

	filename, err := p.filename()

	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	if e, a := shareddefaults.SharedCredentialsFilename(), filename; e != a {
		t.Errorf("expect %q filename, got %q", e, a)
	}
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
