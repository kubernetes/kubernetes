// Package silly provides a simple authentication scheme that checks for the
// existence of an Authorization header and issues access if is present and
// non-empty.
//
// This package is present as an example implementation of a minimal
// auth.AccessController and for testing. This is not suitable for any kind of
// production security.
package silly

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/docker/distribution/context"
	"github.com/docker/distribution/registry/auth"
)

// accessController provides a simple implementation of auth.AccessController
// that simply checks for a non-empty Authorization header. It is useful for
// demonstration and testing.
type accessController struct {
	realm   string
	service string
}

var _ auth.AccessController = &accessController{}

func newAccessController(options map[string]interface{}) (auth.AccessController, error) {
	realm, present := options["realm"]
	if _, ok := realm.(string); !present || !ok {
		return nil, fmt.Errorf(`"realm" must be set for silly access controller`)
	}

	service, present := options["service"]
	if _, ok := service.(string); !present || !ok {
		return nil, fmt.Errorf(`"service" must be set for silly access controller`)
	}

	return &accessController{realm: realm.(string), service: service.(string)}, nil
}

// Authorized simply checks for the existence of the authorization header,
// responding with a bearer challenge if it doesn't exist.
func (ac *accessController) Authorized(ctx context.Context, accessRecords ...auth.Access) (context.Context, error) {
	req, err := context.GetRequest(ctx)
	if err != nil {
		return nil, err
	}

	if req.Header.Get("Authorization") == "" {
		challenge := challenge{
			realm:   ac.realm,
			service: ac.service,
		}

		if len(accessRecords) > 0 {
			var scopes []string
			for _, access := range accessRecords {
				scopes = append(scopes, fmt.Sprintf("%s:%s:%s", access.Type, access.Resource.Name, access.Action))
			}
			challenge.scope = strings.Join(scopes, " ")
		}

		return nil, &challenge
	}

	return auth.WithUser(ctx, auth.UserInfo{Name: "silly"}), nil
}

type challenge struct {
	realm   string
	service string
	scope   string
}

var _ auth.Challenge = challenge{}

// SetHeaders sets a simple bearer challenge on the response.
func (ch challenge) SetHeaders(w http.ResponseWriter) {
	header := fmt.Sprintf("Bearer realm=%q,service=%q", ch.realm, ch.service)

	if ch.scope != "" {
		header = fmt.Sprintf("%s,scope=%q", header, ch.scope)
	}

	w.Header().Set("WWW-Authenticate", header)
}

func (ch challenge) Error() string {
	return fmt.Sprintf("silly authentication challenge: %#v", ch)
}

// init registers the silly auth backend.
func init() {
	auth.Register("silly", auth.InitFunc(newAccessController))
}
