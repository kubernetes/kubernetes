package credentials

import (
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws/awserr"
)

type stubProvider struct {
	creds          Value
	retrievedCount int
	expired        bool
	err            error
}

func (s *stubProvider) Retrieve() (Value, error) {
	s.retrievedCount++
	s.expired = false
	s.creds.ProviderName = "stubProvider"
	return s.creds, s.err
}
func (s *stubProvider) IsExpired() bool {
	return s.expired
}

func TestCredentialsGet(t *testing.T) {
	c := NewCredentials(&stubProvider{
		creds: Value{
			AccessKeyID:     "AKID",
			SecretAccessKey: "SECRET",
			SessionToken:    "",
		},
		expired: true,
	})

	creds, err := c.Get()
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
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

func TestCredentialsGetWithError(t *testing.T) {
	c := NewCredentials(&stubProvider{err: awserr.New("provider error", "", nil), expired: true})

	_, err := c.Get()
	if e, a := "provider error", err.(awserr.Error).Code(); e != a {
		t.Errorf("Expected provider error, %v got %v", e, a)
	}
}

func TestCredentialsExpire(t *testing.T) {
	stub := &stubProvider{}
	c := NewCredentials(stub)

	stub.expired = false
	if !c.IsExpired() {
		t.Errorf("Expected to start out expired")
	}
	c.Expire()
	if !c.IsExpired() {
		t.Errorf("Expected to be expired")
	}

	c.Get()
	if c.IsExpired() {
		t.Errorf("Expected not to be expired")
	}

	stub.expired = true
	if !c.IsExpired() {
		t.Errorf("Expected to be expired")
	}
}

type MockProvider struct {
	Expiry
}

func (*MockProvider) Retrieve() (Value, error) {
	return Value{}, nil
}

func TestCredentialsGetWithProviderName(t *testing.T) {
	stub := &stubProvider{}

	c := NewCredentials(stub)

	creds, err := c.Get()
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if e, a := creds.ProviderName, "stubProvider"; e != a {
		t.Errorf("Expected provider name to match, %v got %v", e, a)
	}
}

func TestCredentialsIsExpired_Race(t *testing.T) {
	creds := NewChainCredentials([]Provider{&MockProvider{}})

	starter := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(10)
	for i := 0; i < 10; i++ {
		go func() {
			defer wg.Done()
			<-starter
			for i := 0; i < 100; i++ {
				creds.IsExpired()
			}
		}()
	}
	close(starter)

	wg.Wait()
}

func TestCredentialsExpiresAt_NoExpirer(t *testing.T) {
	stub := &stubProvider{}
	c := NewCredentials(stub)

	_, err := c.ExpiresAt()
	if e, a := "ProviderNotExpirer", err.(awserr.Error).Code(); e != a {
		t.Errorf("Expected provider error, %v got %v", e, a)
	}
}

type stubProviderExpirer struct {
	stubProvider
	expiration time.Time
}

func (s *stubProviderExpirer) ExpiresAt() time.Time {
	return s.expiration
}

func TestCredentialsExpiresAt_HasExpirer(t *testing.T) {
	stub := &stubProviderExpirer{}
	c := NewCredentials(stub)

	// fetch initial credentials so that forceRefresh is set false
	_, err := c.Get()
	if err != nil {
		t.Errorf("Unexpecte error: %v", err)
	}

	stub.expiration = time.Unix(rand.Int63(), 0)
	expiration, err := c.ExpiresAt()
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if stub.expiration != expiration {
		t.Errorf("Expected matching expiration, %v got %v", stub.expiration, expiration)
	}

	c.Expire()
	expiration, err = c.ExpiresAt()
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if !expiration.IsZero() {
		t.Errorf("Expected distant past expiration, got %v", expiration)
	}
}

type stubProviderConcurrent struct {
	stubProvider
	done chan struct{}
}

func (s *stubProviderConcurrent) Retrieve() (Value, error) {
	<-s.done
	return s.stubProvider.Retrieve()
}

func TestCredentialsGetConcurrent(t *testing.T) {
	stub := &stubProviderConcurrent{
		done: make(chan struct{}),
	}

	c := NewCredentials(stub)
	done := make(chan struct{})

	for i := 0; i < 2; i++ {
		go func() {
			c.Get()
			done <- struct{}{}
		}()
	}

	// Validates that a single call to Retrieve is shared between two calls to Get
	stub.done <- struct{}{}
	<-done
	<-done
}

type stubProviderRefreshable struct {
	creds        Value
	expired      bool
	hasRetrieved bool
}

func (s *stubProviderRefreshable) Retrieve() (Value, error) {
	// On first retrieval, return the creds that this provider was created with.
	// On subsequent retrievals, return new refreshed credentials.
	if !s.hasRetrieved {
		s.expired = true
		s.hasRetrieved = true
	} else {
		s.creds = Value{
			AccessKeyID:     "AKID",
			SecretAccessKey: "SECRET",
			SessionToken:    "NEW_SESSION",
		}
		s.expired = false
		time.Sleep(10 * time.Millisecond)
	}
	return s.creds, nil
}

func (s *stubProviderRefreshable) IsExpired() bool {
	return s.expired
}

func TestCredentialsGet_RefreshableProviderRace(t *testing.T) {
	stub := &stubProviderRefreshable{
		creds: Value{
			AccessKeyID:     "AKID",
			SecretAccessKey: "SECRET",
			SessionToken:    "OLD_SESSION",
		},
	}

	c := NewCredentials(stub)

	// The first Get() causes stubProviderRefreshable to consider its
	// OLD_SESSION credentials expired on subsequent retrievals.
	creds, err := c.Get()
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if e, a := "OLD_SESSION", creds.SessionToken; e != a {
		t.Errorf("Expect session token to match, %v got %v", e, a)
	}

	// Since stubProviderRefreshable considers its OLD_SESSION credentials
	// expired, all subsequent calls to Get() should retrieve NEW_SESSION creds.
	var wg sync.WaitGroup
	wg.Add(100)
	for i := 0; i < 100; i++ {
		go func() {
			defer wg.Done()
			creds, err := c.Get()
			if err != nil {
				t.Errorf("Expected no error, got %v", err)
			}

			if c.IsExpired() {
				t.Errorf("not expect expired")
			}

			if e, a := "NEW_SESSION", creds.SessionToken; e != a {
				t.Errorf("Expect session token to match, %v got %v", e, a)
			}
		}()
	}

	wg.Wait()
}
