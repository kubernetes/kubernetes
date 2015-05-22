package credentials

import (
	"fmt"
	"os"
)

var (
	// ErrAccessKeyIDNotFound is returned when the AWS Access Key ID can't be
	// found in the process's environment.
	ErrAccessKeyIDNotFound = fmt.Errorf("AWS_ACCESS_KEY_ID or AWS_ACCESS_KEY not found in environment")
	// ErrSecretAccessKeyNotFound is returned when the AWS Secret Access Key
	// can't be found in the process's environment.
	ErrSecretAccessKeyNotFound = fmt.Errorf("AWS_SECRET_ACCESS_KEY or AWS_SECRET_KEY not found in environment")
)

// A EnvProvider retrieves credentials from the environment variables of the
// running process. Environment credentials never expire.
//
// Environment variables used:
// - Access Key ID:     AWS_ACCESS_KEY_ID or AWS_ACCESS_KEY
// - Secret Access Key: AWS_SECRET_ACCESS_KEY or AWS_SECRET_KEY
type EnvProvider struct {
	retrieved bool
}

// NewEnvCredentials returns a pointer to a new Credentials object
// wrapping the environment variable provider.
func NewEnvCredentials() *Credentials {
	return NewCredentials(&EnvProvider{})
}

// Retrieve retrieves the keys from the environment.
func (e *EnvProvider) Retrieve() (Value, error) {
	e.retrieved = false

	id := os.Getenv("AWS_ACCESS_KEY_ID")
	if id == "" {
		id = os.Getenv("AWS_ACCESS_KEY")
	}

	secret := os.Getenv("AWS_SECRET_ACCESS_KEY")
	if secret == "" {
		secret = os.Getenv("AWS_SECRET_KEY")
	}

	if id == "" {
		return Value{}, ErrAccessKeyIDNotFound
	}

	if secret == "" {
		return Value{}, ErrSecretAccessKeyNotFound
	}

	e.retrieved = true
	return Value{
		AccessKeyID:     id,
		SecretAccessKey: secret,
		SessionToken:    os.Getenv("AWS_SESSION_TOKEN"),
	}, nil
}

// IsExpired returns if the credentials have been retrieved.
func (e *EnvProvider) IsExpired() bool {
	return !e.retrieved
}
