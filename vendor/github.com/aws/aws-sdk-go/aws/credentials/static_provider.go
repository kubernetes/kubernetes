package credentials

import (
	"github.com/aws/aws-sdk-go/aws/awserr"
)

// StaticProviderName provides a name of Static provider
const StaticProviderName = "StaticProvider"

var (
	// ErrStaticCredentialsEmpty is emitted when static credentials are empty.
	//
	// @readonly
	ErrStaticCredentialsEmpty = awserr.New("EmptyStaticCreds", "static credentials are empty", nil)
)

// A StaticProvider is a set of credentials which are set programmatically,
// and will never expire.
type StaticProvider struct {
	Value
}

// NewStaticCredentials returns a pointer to a new Credentials object
// wrapping a static credentials value provider.
func NewStaticCredentials(id, secret, token string) *Credentials {
	return NewCredentials(&StaticProvider{Value: Value{
		AccessKeyID:     id,
		SecretAccessKey: secret,
		SessionToken:    token,
	}})
}

// Retrieve returns the credentials or error if the credentials are invalid.
func (s *StaticProvider) Retrieve() (Value, error) {
	if s.AccessKeyID == "" || s.SecretAccessKey == "" {
		return Value{ProviderName: StaticProviderName}, ErrStaticCredentialsEmpty
	}

	s.Value.ProviderName = StaticProviderName
	return s.Value, nil
}

// IsExpired returns if the credentials are expired.
//
// For StaticProvider, the credentials never expired.
func (s *StaticProvider) IsExpired() bool {
	return false
}
