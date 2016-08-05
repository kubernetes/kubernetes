package auth

import (
	"errors"
	"fmt"
	"sync"

	log "github.com/golang/glog"
	"github.com/mesos/mesos-go/auth/callback"
	"golang.org/x/net/context"
)

// SPI interface: login provider implementations support this interface, clients
// do not authenticate against this directly, instead they should use Login()
type Authenticatee interface {
	// Returns no errors if successfully authenticated, otherwise a single
	// error.
	Authenticate(ctx context.Context, handler callback.Handler) error
}

// Func adapter for interface: allow func's to implement the Authenticatee interface
// as long as the func signature matches
type AuthenticateeFunc func(ctx context.Context, handler callback.Handler) error

func (f AuthenticateeFunc) Authenticate(ctx context.Context, handler callback.Handler) error {
	return f(ctx, handler)
}

var (
	// Authentication was attempted and failed (likely due to incorrect credentials, too
	// many retries within a time window, etc). Distinctly different from authentication
	// errors (e.g. network errors, configuration errors, etc).
	AuthenticationFailed = errors.New("authentication failed")

	authenticateeProviders = make(map[string]Authenticatee) // authentication providers dict
	providerLock           sync.Mutex
)

// Register an authentication provider (aka "login provider"). packages that
// provide Authenticatee implementations should invoke this func in their
// init() to register.
func RegisterAuthenticateeProvider(name string, auth Authenticatee) (err error) {
	providerLock.Lock()
	defer providerLock.Unlock()

	if _, found := authenticateeProviders[name]; found {
		err = fmt.Errorf("authentication provider already registered: %v", name)
	} else {
		authenticateeProviders[name] = auth
		log.V(1).Infof("registered authentication provider: %v", name)
	}
	return
}

// Look up an authentication provider by name, returns non-nil and true if such
// a provider is found.
func getAuthenticateeProvider(name string) (provider Authenticatee, ok bool) {
	providerLock.Lock()
	defer providerLock.Unlock()

	provider, ok = authenticateeProviders[name]
	return
}
