// Package htpasswd provides a simple authentication scheme that checks for the
// user credential hash in an htpasswd formatted file in a configuration-determined
// location.
//
// This authentication method MUST be used under TLS, as simple token-replay attack is possible.
package htpasswd

import (
	"fmt"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/docker/distribution/context"
	"github.com/docker/distribution/registry/auth"
)

type accessController struct {
	realm    string
	path     string
	modtime  time.Time
	mu       sync.Mutex
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

	return &accessController{realm: realm.(string), path: path.(string)}, nil
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
			err:   auth.ErrInvalidCredential,
		}
	}

	// Dynamically parsing the latest account list
	fstat, err := os.Stat(ac.path)
	if err != nil {
		return nil, err
	}

	lastModified := fstat.ModTime()
	ac.mu.Lock()
	if ac.htpasswd == nil || !ac.modtime.Equal(lastModified) {
		ac.modtime = lastModified

		f, err := os.Open(ac.path)
		if err != nil {
			ac.mu.Unlock()
			return nil, err
		}
		defer f.Close()

		h, err := newHTPasswd(f)
		if err != nil {
			ac.mu.Unlock()
			return nil, err
		}
		ac.htpasswd = h
	}
	localHTPasswd := ac.htpasswd
	ac.mu.Unlock()

	if err := localHTPasswd.authenticateUser(username, password); err != nil {
		context.GetLogger(ctx).Errorf("error authenticating user %q: %v", username, err)
		return nil, &challenge{
			realm: ac.realm,
			err:   auth.ErrAuthenticationFailure,
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
