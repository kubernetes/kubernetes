// Package htpasswd provides a simple authentication scheme that checks for the
// user credential hash in an htpasswd formatted file in a configuration-determined
// location.
//
// This authentication method MUST be used under TLS, as simple token-replay attack is possible.
package htpasswd

import (
	"errors"
	"fmt"
	"net/http"
	"os"

	"github.com/docker/distribution/context"
	"github.com/docker/distribution/registry/auth"
)

var (
	// ErrInvalidCredential is returned when the auth token does not authenticate correctly.
	ErrInvalidCredential = errors.New("invalid authorization credential")

	// ErrAuthenticationFailure returned when authentication failure to be presented to agent.
	ErrAuthenticationFailure = errors.New("authentication failure")
)

type accessController struct {
	realm    string
	htpasswd *htpasswd
}

var _ auth.AccessController = &accessController{}

func newAccessController(options map[string]interface{}) (auth.AccessController, error) {
	realm, present := options["realm"]
	if _, ok := realm.(string); !present || !ok {
		return nil, fmt.Errorf(`"realm" must be set for htpasswd access controller`)
	}

	path, present := options["path"]
	if _, ok := path.(string); !present || !ok {
		return nil, fmt.Errorf(`"path" must be set for htpasswd access controller`)
	}

	f, err := os.Open(path.(string))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	h, err := newHTPasswd(f)
	if err != nil {
		return nil, err
	}

	return &accessController{realm: realm.(string), htpasswd: h}, nil
}

func (ac *accessController) Authorized(ctx context.Context, accessRecords ...auth.Access) (context.Context, error) {
	req, err := context.GetRequest(ctx)
	if err != nil {
		return nil, err
	}

	username, password, ok := req.BasicAuth()
	if !ok {
		return nil, &challenge{
			realm: ac.realm,
			err:   ErrInvalidCredential,
		}
	}

	if err := ac.htpasswd.authenticateUser(username, password); err != nil {
		context.GetLogger(ctx).Errorf("error authenticating user %q: %v", username, err)
		return nil, &challenge{
			realm: ac.realm,
			err:   ErrAuthenticationFailure,
		}
	}

	return auth.WithUser(ctx, auth.UserInfo{Name: username}), nil
}

// challenge implements the auth.Challenge interface.
type challenge struct {
	realm string
	err   error
}

var _ auth.Challenge = challenge{}

// SetHeaders sets the basic challenge header on the response.
func (ch challenge) SetHeaders(w http.ResponseWriter) {
	w.Header().Set("WWW-Authenticate", fmt.Sprintf("Basic realm=%q", ch.realm))
}

func (ch challenge) Error() string {
	return fmt.Sprintf("basic authentication challenge for realm %q: %s", ch.realm, ch.err)
}

func init() {
	auth.Register("htpasswd", auth.InitFunc(newAccessController))
}
