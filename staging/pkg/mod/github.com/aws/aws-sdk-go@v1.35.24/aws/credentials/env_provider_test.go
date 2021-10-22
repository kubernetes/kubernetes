package credentials

import (
	"os"
	"testing"

	"github.com/aws/aws-sdk-go/internal/sdktesting"
)

func TestEnvProviderRetrieve(t *testing.T) {
	restoreEnvFn := sdktesting.StashEnv()
	defer restoreEnvFn()
	os.Setenv("AWS_ACCESS_KEY_ID", "access")
	os.Setenv("AWS_SECRET_ACCESS_KEY", "secret")
	os.Setenv("AWS_SESSION_TOKEN", "token")

	e := EnvProvider{}
	creds, err := e.Retrieve()
	if err != nil {
		t.Errorf("expect nil, got %v", err)
	}

	if e, a := "access", creds.AccessKeyID; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "secret", creds.SecretAccessKey; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "token", creds.SessionToken; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestEnvProviderIsExpired(t *testing.T) {
	restoreEnvFn := sdktesting.StashEnv()
	defer restoreEnvFn()

	os.Setenv("AWS_ACCESS_KEY_ID", "access")
	os.Setenv("AWS_SECRET_ACCESS_KEY", "secret")
	os.Setenv("AWS_SESSION_TOKEN", "token")

	e := EnvProvider{}

	if !e.IsExpired() {
		t.Errorf("Expect creds to be expired before retrieve.")
	}

	_, err := e.Retrieve()
	if err != nil {
		t.Errorf("expect nil, got %v", err)
	}

	if e.IsExpired() {
		t.Errorf("Expect creds to not be expired after retrieve.")
	}
}

func TestEnvProviderNoAccessKeyID(t *testing.T) {
	restoreEnvFn := sdktesting.StashEnv()
	defer restoreEnvFn()

	os.Setenv("AWS_SECRET_ACCESS_KEY", "secret")

	e := EnvProvider{}
	_, err := e.Retrieve()
	if e, a := ErrAccessKeyIDNotFound, err; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestEnvProviderNoSecretAccessKey(t *testing.T) {
	restoreEnvFn := sdktesting.StashEnv()
	defer restoreEnvFn()

	os.Setenv("AWS_ACCESS_KEY_ID", "access")

	e := EnvProvider{}
	_, err := e.Retrieve()
	if e, a := ErrSecretAccessKeyNotFound, err; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestEnvProviderAlternateNames(t *testing.T) {
	restoreEnvFn := sdktesting.StashEnv()
	defer restoreEnvFn()

	os.Setenv("AWS_ACCESS_KEY", "access")
	os.Setenv("AWS_SECRET_KEY", "secret")

	e := EnvProvider{}
	creds, err := e.Retrieve()
	if err != nil {
		t.Errorf("expect nil, got %v", err)
	}

	if e, a := "access", creds.AccessKeyID; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "secret", creds.SecretAccessKey; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if v := creds.SessionToken; len(v) != 0 {
		t.Errorf("Expected no token, %v", v)
	}
}
