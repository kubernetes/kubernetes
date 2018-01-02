package credentials

import (
	"testing"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/stretchr/testify/assert"
)

type secondStubProvider struct {
	creds   Value
	expired bool
	err     error
}

func (s *secondStubProvider) Retrieve() (Value, error) {
	s.expired = false
	s.creds.ProviderName = "secondStubProvider"
	return s.creds, s.err
}
func (s *secondStubProvider) IsExpired() bool {
	return s.expired
}

func TestChainProviderWithNames(t *testing.T) {
	p := &ChainProvider{
		Providers: []Provider{
			&stubProvider{err: awserr.New("FirstError", "first provider error", nil)},
			&stubProvider{err: awserr.New("SecondError", "second provider error", nil)},
			&secondStubProvider{
				creds: Value{
					AccessKeyID:     "AKIF",
					SecretAccessKey: "NOSECRET",
					SessionToken:    "",
				},
			},
			&stubProvider{
				creds: Value{
					AccessKeyID:     "AKID",
					SecretAccessKey: "SECRET",
					SessionToken:    "",
				},
			},
		},
	}

	creds, err := p.Retrieve()
	assert.Nil(t, err, "Expect no error")
	assert.Equal(t, "secondStubProvider", creds.ProviderName, "Expect provider name to match")

	// Also check credentials
	assert.Equal(t, "AKIF", creds.AccessKeyID, "Expect access key ID to match")
	assert.Equal(t, "NOSECRET", creds.SecretAccessKey, "Expect secret access key to match")
	assert.Empty(t, creds.SessionToken, "Expect session token to be empty")

}

func TestChainProviderGet(t *testing.T) {
	p := &ChainProvider{
		Providers: []Provider{
			&stubProvider{err: awserr.New("FirstError", "first provider error", nil)},
			&stubProvider{err: awserr.New("SecondError", "second provider error", nil)},
			&stubProvider{
				creds: Value{
					AccessKeyID:     "AKID",
					SecretAccessKey: "SECRET",
					SessionToken:    "",
				},
			},
		},
	}

	creds, err := p.Retrieve()
	assert.Nil(t, err, "Expect no error")
	assert.Equal(t, "AKID", creds.AccessKeyID, "Expect access key ID to match")
	assert.Equal(t, "SECRET", creds.SecretAccessKey, "Expect secret access key to match")
	assert.Empty(t, creds.SessionToken, "Expect session token to be empty")
}

func TestChainProviderIsExpired(t *testing.T) {
	stubProvider := &stubProvider{expired: true}
	p := &ChainProvider{
		Providers: []Provider{
			stubProvider,
		},
	}

	assert.True(t, p.IsExpired(), "Expect expired to be true before any Retrieve")
	_, err := p.Retrieve()
	assert.Nil(t, err, "Expect no error")
	assert.False(t, p.IsExpired(), "Expect not expired after retrieve")

	stubProvider.expired = true
	assert.True(t, p.IsExpired(), "Expect return of expired provider")

	_, err = p.Retrieve()
	assert.False(t, p.IsExpired(), "Expect not expired after retrieve")
}

func TestChainProviderWithNoProvider(t *testing.T) {
	p := &ChainProvider{
		Providers: []Provider{},
	}

	assert.True(t, p.IsExpired(), "Expect expired with no providers")
	_, err := p.Retrieve()
	assert.Equal(t,
		ErrNoValidProvidersFoundInChain,
		err,
		"Expect no providers error returned")
}

func TestChainProviderWithNoValidProvider(t *testing.T) {
	errs := []error{
		awserr.New("FirstError", "first provider error", nil),
		awserr.New("SecondError", "second provider error", nil),
	}
	p := &ChainProvider{
		Providers: []Provider{
			&stubProvider{err: errs[0]},
			&stubProvider{err: errs[1]},
		},
	}

	assert.True(t, p.IsExpired(), "Expect expired with no providers")
	_, err := p.Retrieve()

	assert.Equal(t,
		ErrNoValidProvidersFoundInChain,
		err,
		"Expect no providers error returned")
}

func TestChainProviderWithNoValidProviderWithVerboseEnabled(t *testing.T) {
	errs := []error{
		awserr.New("FirstError", "first provider error", nil),
		awserr.New("SecondError", "second provider error", nil),
	}
	p := &ChainProvider{
		VerboseErrors: true,
		Providers: []Provider{
			&stubProvider{err: errs[0]},
			&stubProvider{err: errs[1]},
		},
	}

	assert.True(t, p.IsExpired(), "Expect expired with no providers")
	_, err := p.Retrieve()

	assert.Equal(t,
		awserr.NewBatchError("NoCredentialProviders", "no valid providers in chain", errs),
		err,
		"Expect no providers error returned")
}
