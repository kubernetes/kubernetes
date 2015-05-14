package credentials

import (
	"errors"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestChainProviderGet(t *testing.T) {
	p := &ChainProvider{
		Providers: []Provider{
			&stubProvider{err: errors.New("first provider error")},
			&stubProvider{err: errors.New("second provider error")},
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
	assert.Equal(t, ErrNoValidProvidersFoundInChain, err, "Expect no providers error returned")
}

func TestChainProviderWithNoValidProvider(t *testing.T) {
	p := &ChainProvider{
		Providers: []Provider{
			&stubProvider{err: errors.New("first provider error")},
			&stubProvider{err: errors.New("second provider error")},
		},
	}

	assert.True(t, p.IsExpired(), "Expect expired with no providers")
	_, err := p.Retrieve()
	assert.Contains(t, "no valid providers in chain", err.Error(), "Expect no providers error returned")
}
