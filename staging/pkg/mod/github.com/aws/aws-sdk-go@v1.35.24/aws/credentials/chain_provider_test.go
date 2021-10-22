package credentials

import (
	"reflect"
	"testing"

	"github.com/aws/aws-sdk-go/aws/awserr"
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
	if err != nil {
		t.Errorf("Expect no error, got %v", err)
	}
	if e, a := "secondStubProvider", creds.ProviderName; e != a {
		t.Errorf("Expect provider name to match, %v got, %v", e, a)
	}

	// Also check credentials
	if e, a := "AKIF", creds.AccessKeyID; e != a {
		t.Errorf("Expect access key ID to match, %v got %v", e, a)
	}
	if e, a := "NOSECRET", creds.SecretAccessKey; e != a {
		t.Errorf("Expect secret access key to match, %v got %v", e, a)
	}
	if v := creds.SessionToken; len(v) != 0 {
		t.Errorf("Expect session token to be empty, %v", v)
	}

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
	if err != nil {
		t.Errorf("Expect no error, got %v", err)
	}
	if e, a := "AKID", creds.AccessKeyID; e != a {
		t.Errorf("Expect access key ID to match, %v got %v", e, a)
	}
	if e, a := "SECRET", creds.SecretAccessKey; e != a {
		t.Errorf("Expect secret access key to match, %v got %v", e, a)
	}
	if v := creds.SessionToken; len(v) != 0 {
		t.Errorf("Expect session token to be empty, %v", v)
	}
}

func TestChainProviderIsExpired(t *testing.T) {
	stubProvider := &stubProvider{expired: true}
	p := &ChainProvider{
		Providers: []Provider{
			stubProvider,
		},
	}

	if !p.IsExpired() {
		t.Errorf("Expect expired to be true before any Retrieve")
	}
	_, err := p.Retrieve()
	if err != nil {
		t.Errorf("Expect no error, got %v", err)
	}
	if p.IsExpired() {
		t.Errorf("Expect not expired after retrieve")
	}

	stubProvider.expired = true
	if !p.IsExpired() {
		t.Errorf("Expect return of expired provider")
	}

	_, err = p.Retrieve()
	if err != nil {
		t.Errorf("Expect no error, got %v", err)
	}
	if p.IsExpired() {
		t.Errorf("Expect not expired after retrieve")
	}
}

func TestChainProviderWithNoProvider(t *testing.T) {
	p := &ChainProvider{
		Providers: []Provider{},
	}

	if !p.IsExpired() {
		t.Errorf("Expect expired with no providers")
	}
	_, err := p.Retrieve()
	if e, a := ErrNoValidProvidersFoundInChain, err; e != a {
		t.Errorf("Expect no providers error returned, %v, got %v", e, a)
	}
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

	if !p.IsExpired() {
		t.Errorf("Expect expired with no providers")
	}
	_, err := p.Retrieve()

	if e, a := ErrNoValidProvidersFoundInChain, err; e != a {
		t.Errorf("Expect no providers error returned, %v, got %v", e, a)
	}
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

	if !p.IsExpired() {
		t.Errorf("Expect expired with no providers")
	}
	_, err := p.Retrieve()

	expectErr := awserr.NewBatchError("NoCredentialProviders", "no valid providers in chain", errs)
	if e, a := expectErr, err; !reflect.DeepEqual(e, a) {
		t.Errorf("Expect no providers error returned, %v, got %v", e, a)
	}
}
