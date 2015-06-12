package auth

import (
	"errors"
	"fmt"

	"github.com/mesos/mesos-go/auth/callback"
	"github.com/mesos/mesos-go/upid"
	"golang.org/x/net/context"
)

var (
	// No login provider name has been specified in a context.Context
	NoLoginProviderName = errors.New("missing login provider name in context")
)

// Main client entrypoint into the authentication APIs: clients are expected to
// invoke this func with a context containing a login provider name value.
// This may be written as:
//     providerName := ...  // the user has probably configured this via some flag
//     handler := ...  // handlers provide data like usernames and passwords
//     ctx := ...  // obtain some initial or timed context
//     err := auth.Login(auth.WithLoginProvider(ctx, providerName), handler)
func Login(ctx context.Context, handler callback.Handler) error {
	name, ok := LoginProviderFrom(ctx)
	if !ok {
		return NoLoginProviderName
	}
	provider, ok := getAuthenticateeProvider(name)
	if !ok {
		return fmt.Errorf("unrecognized login provider name in context: %s", name)
	}
	return provider.Authenticate(ctx, handler)
}

// Unexported key type, avoids conflicts with other context-using packages. All
// context items registered from this package should use keys of this type.
type loginKeyType int

const (
	loginProviderNameKey loginKeyType = iota // name of login provider to use
	parentUpidKey                            // upid.UPID of some parent process
)

// Return a context that inherits all values from the parent ctx and specifies
// the login provider name given here. Intended to be invoked before calls to
// Login().
func WithLoginProvider(ctx context.Context, providerName string) context.Context {
	return context.WithValue(ctx, loginProviderNameKey, providerName)
}

// Return the name of the login provider specified in this context.
func LoginProviderFrom(ctx context.Context) (name string, ok bool) {
	name, ok = ctx.Value(loginProviderNameKey).(string)
	return
}

// Return the name of the login provider specified in this context, or empty
// string if none.
func LoginProvider(ctx context.Context) string {
	name, _ := LoginProviderFrom(ctx)
	return name
}

func WithParentUPID(ctx context.Context, pid upid.UPID) context.Context {
	return context.WithValue(ctx, parentUpidKey, pid)
}

func ParentUPIDFrom(ctx context.Context) (pid upid.UPID, ok bool) {
	pid, ok = ctx.Value(parentUpidKey).(upid.UPID)
	return
}

func ParentUPID(ctx context.Context) (upid *upid.UPID) {
	if upid, ok := ParentUPIDFrom(ctx); ok {
		return &upid
	} else {
		return nil
	}
}
