package credentials

import (
	"testing"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/stretchr/testify/assert"
)

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
	assert.Equal(t, ErrNoValidProvidersFoundInChain, err, "Expect no providers error returned")
}

func TestChainProviderWithNoValidProvider(t *testing.T) {
	p := &ChainProvider{
		Providers: []Provider{
			&stubProvider{err: awserr.New("FirstError", "first provider error", nil)},
			&stubProvider{err: awserr.New("SecondError", "second provider error", nil)},
		},
	}

	assert.True(t, p.IsExpired(), "Expect expired with no providers")
	_, err := p.Retrieve()
	assert.Equal(t, ErrNoValidProvidersFoundInChain, err, "Expect no providers error returned")
}
