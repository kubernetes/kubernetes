package credentials

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/aws/aws-sdk-go/internal/sdktesting"
	"github.com/aws/aws-sdk-go/internal/shareddefaults"
)

func TestSharedCredentialsProvider(t *testing.T) {
	restoreEnvFn := sdktesting.StashEnv()
	defer restoreEnvFn()

	p := SharedCredentialsProvider{Filename: "example.ini", Profile: ""}
	creds, err := p.Retrieve()
	if err != nil {
		t.Errorf("expect nil, got %v", err)
	}

	if e, a := "accessKey", creds.AccessKeyID; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "secret", creds.SecretAccessKey; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "token", creds.SessionToken; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestSharedCredentialsProviderIsExpired(t *testing.T) {
	restoreEnvFn := sdktesting.StashEnv()
	defer restoreEnvFn()

	p := SharedCredentialsProvider{Filename: "example.ini", Profile: ""}

	if !p.IsExpired() {
		t.Errorf("Expect creds to be expired before retrieve")
	}

	_, err := p.Retrieve()
	if err != nil {
		t.Errorf("expect nil, got %v", err)
	}

	if p.IsExpired() {
		t.Errorf("Expect creds to not be expired after retrieve")
	}
}

func TestSharedCredentialsProviderWithAWS_SHARED_CREDENTIALS_FILE(t *testing.T) {
	restoreEnvFn := sdktesting.StashEnv()
	defer restoreEnvFn()

	os.Setenv("AWS_SHARED_CREDENTIALS_FILE", "example.ini")
	p := SharedCredentialsProvider{}
	creds, err := p.Retrieve()

	if err != nil {
		t.Errorf("expect nil, got %v", err)
	}

	if e, a := "accessKey", creds.AccessKeyID; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "secret", creds.SecretAccessKey; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "token", creds.SessionToken; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestSharedCredentialsProviderWithAWS_SHARED_CREDENTIALS_FILEAbsPath(t *testing.T) {
	restoreEnvFn := sdktesting.StashEnv()
	defer restoreEnvFn()

	wd, err := os.Getwd()
	if err != nil {
		t.Errorf("expect no error, got %v", err)
	}
	os.Setenv("AWS_SHARED_CREDENTIALS_FILE", filepath.Join(wd, "example.ini"))
	p := SharedCredentialsProvider{}
	creds, err := p.Retrieve()
	if err != nil {
		t.Errorf("expect nil, got %v", err)
	}

	if e, a := "accessKey", creds.AccessKeyID; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "secret", creds.SecretAccessKey; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "token", creds.SessionToken; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestSharedCredentialsProviderWithAWS_PROFILE(t *testing.T) {
	restoreEnvFn := sdktesting.StashEnv()
	defer restoreEnvFn()

	os.Setenv("AWS_PROFILE", "no_token")

	p := SharedCredentialsProvider{Filename: "example.ini", Profile: ""}
	creds, err := p.Retrieve()
	if err != nil {
		t.Errorf("expect nil, got %v", err)
	}

	if e, a := "accessKey", creds.AccessKeyID; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "secret", creds.SecretAccessKey; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if v := creds.SessionToken; len(v) != 0 {
		t.Errorf("Expect no token, %v", v)
	}
}

func TestSharedCredentialsProviderWithoutTokenFromProfile(t *testing.T) {
	restoreEnvFn := sdktesting.StashEnv()
	defer restoreEnvFn()

	p := SharedCredentialsProvider{Filename: "example.ini", Profile: "no_token"}
	creds, err := p.Retrieve()
	if err != nil {
		t.Errorf("expect nil, got %v", err)
	}

	if e, a := "accessKey", creds.AccessKeyID; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "secret", creds.SecretAccessKey; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if v := creds.SessionToken; len(v) != 0 {
		t.Errorf("Expect no token, %v", v)
	}
}

func TestSharedCredentialsProviderColonInCredFile(t *testing.T) {
	restoreEnvFn := sdktesting.StashEnv()
	defer restoreEnvFn()

	p := SharedCredentialsProvider{Filename: "example.ini", Profile: "with_colon"}
	creds, err := p.Retrieve()
	if err != nil {
		t.Errorf("expect nil, got %v", err)
	}

	if e, a := "accessKey", creds.AccessKeyID; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "secret", creds.SecretAccessKey; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if v := creds.SessionToken; len(v) != 0 {
		t.Errorf("Expect no token, %v", v)
	}
}

func TestSharedCredentialsProvider_DefaultFilename(t *testing.T) {
	restoreEnvFn := sdktesting.StashEnv()
	defer restoreEnvFn()

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
	restoreEnvFn := sdktesting.StashEnv()
	defer restoreEnvFn()

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
