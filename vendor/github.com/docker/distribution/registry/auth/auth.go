// Package auth defines a standard interface for request access controllers.
//
// An access controller has a simple interface with a single `Authorized`
// method which checks that a given request is authorized to perform one or
// more actions on one or more resources. This method should return a non-nil
// error if the request is not authorized.
//
// An implementation registers its access controller by name with a constructor
// which accepts an options map for configuring the access controller.
//
//		options := map[string]interface{}{"sillySecret": "whysosilly?"}
// 		accessController, _ := auth.GetAccessController("silly", options)
//
// This `accessController` can then be used in a request handler like so:
//
// 		func updateOrder(w http.ResponseWriter, r *http.Request) {
//			orderNumber := r.FormValue("orderNumber")
//			resource := auth.Resource{Type: "customerOrder", Name: orderNumber}
// 			access := auth.Access{Resource: resource, Action: "update"}
//
// 			if ctx, err := accessController.Authorized(ctx, access); err != nil {
//				if challenge, ok := err.(auth.Challenge) {
//					// Let the challenge write the response.
//					challenge.SetHeaders(w)
//					w.WriteHeader(http.StatusUnauthorized)
//					return
//				} else {
//					// Some other error.
//				}
//			}
// 		}
//
package auth

import (
	"errors"
	"fmt"
	"net/http"

	"github.com/docker/distribution/context"
)

const (
	// UserKey is used to get the user object from
	// a user context
	UserKey = "auth.user"

	// UserNameKey is used to get the user name from
	// a user context
	UserNameKey = "auth.user.name"
)

var (
	// ErrInvalidCredential is returned when the auth token does not authenticate correctly.
	ErrInvalidCredential = errors.New("invalid authorization credential")

	// ErrAuthenticationFailure returned when authentication fails.
	ErrAuthenticationFailure = errors.New("authentication failure")
)

// UserInfo carries information about
// an autenticated/authorized client.
type UserInfo struct {
	Name string
}

// Resource describes a resource by type and name.
type Resource struct {
	Type  string
	Class string
	Name  string
}

// Access describes a specific action that is
// requested or allowed for a given resource.
type Access struct {
	Resource
	Action string
}

// Challenge is a special error type which is used for HTTP 401 Unauthorized
// responses and is able to write the response with WWW-Authenticate challenge
// header values based on the error.
type Challenge interface {
	error

	// SetHeaders prepares the request to conduct a challenge response by
	// adding the an HTTP challenge header on the response message. Callers
	// are expected to set the appropriate HTTP status code (e.g. 401)
	// themselves.
	SetHeaders(w http.ResponseWriter)
}

// AccessController controls access to registry resources based on a request
// and required access levels for a request. Implementations can support both
// complete denial and http authorization challenges.
type AccessController interface {
	// Authorized returns a non-nil error if the context is granted access and
	// returns a new authorized context. If one or more Access structs are
	// provided, the requested access will be compared with what is available
	// to the context. The given context will contain a "http.request" key with
	// a `*http.Request` value. If the error is non-nil, access should always
	// be denied. The error may be of type Challenge, in which case the caller
	// may have the Challenge handle the request or choose what action to take
	// based on the Challenge header or response status. The returned context
	// object should have a "auth.user" value set to a UserInfo struct.
	Authorized(ctx context.Context, access ...Access) (context.Context, error)
}

// CredentialAuthenticator is an object which is able to authenticate credentials
type CredentialAuthenticator interface {
	AuthenticateUser(username, password string) error
}

// WithUser returns a context with the authorized user info.
func WithUser(ctx context.Context, user UserInfo) context.Context {
	return userInfoContext{
		Context: ctx,
		user:    user,
	}
}

type userInfoContext struct {
	context.Context
	user UserInfo
}

func (uic userInfoContext) Value(key interface{}) interface{} {
	switch key {
	case UserKey:
		return uic.user
	case UserNameKey:
		return uic.user.Name
	}

	return uic.Context.Value(key)
}

// WithResources returns a context with the authorized resources.
func WithResources(ctx context.Context, resources []Resource) context.Context {
	return resourceContext{
		Context:   ctx,
		resources: resources,
	}
}

type resourceContext struct {
	context.Context
	resources []Resource
}

type resourceKey struct{}

func (rc resourceContext) Value(key interface{}) interface{} {
	if key == (resourceKey{}) {
		return rc.resources
	}

	return rc.Context.Value(key)
}

// AuthorizedResources returns the list of resources which have
// been authorized for this request.
func AuthorizedResources(ctx context.Context) []Resource {
	if resources, ok := ctx.Value(resourceKey{}).([]Resource); ok {
		return resources
	}

	return nil
}

// InitFunc is the type of an AccessController factory function and is used
// to register the constructor for different AccesController backends.
type InitFunc func(options map[string]interface{}) (AccessController, error)

var accessControllers map[string]InitFunc

func init() {
	accessControllers = make(map[string]InitFunc)
}

// Register is used to register an InitFunc for
// an AccessController backend with the given name.
func Register(name string, initFunc InitFunc) error {
	if _, exists := accessControllers[name]; exists {
		return fmt.Errorf("name already registered: %s", name)
	}

	accessControllers[name] = initFunc

	return nil
}

// GetAccessController constructs an AccessController
// with the given options using the named backend.
func GetAccessController(name string, options map[string]interface{}) (AccessController, error) {
	if initFunc, exists := accessControllers[name]; exists {
		return initFunc(options)
	}

	return nil, fmt.Errorf("no access controller registered with name: %s", name)
}
