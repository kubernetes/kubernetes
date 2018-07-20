package router

import (
	"net/http"

	"golang.org/x/net/context"

	"github.com/docker/docker/api/server/httputils"
)

// ExperimentalRoute defines an experimental API route that can be enabled or disabled.
type ExperimentalRoute interface {
	Route

	Enable()
	Disable()
}

// experimentalRoute defines an experimental API route that can be enabled or disabled.
// It implements ExperimentalRoute
type experimentalRoute struct {
	local   Route
	handler httputils.APIFunc
}

// Enable enables this experimental route
func (r *experimentalRoute) Enable() {
	r.handler = r.local.Handler()
}

// Disable disables the experimental route
func (r *experimentalRoute) Disable() {
	r.handler = experimentalHandler
}

type notImplementedError struct{}

func (notImplementedError) Error() string {
	return "This experimental feature is disabled by default. Start the Docker daemon in experimental mode in order to enable it."
}

func (notImplementedError) NotImplemented() {}

func experimentalHandler(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	return notImplementedError{}
}

// Handler returns returns the APIFunc to let the server wrap it in middlewares.
func (r *experimentalRoute) Handler() httputils.APIFunc {
	return r.handler
}

// Method returns the http method that the route responds to.
func (r *experimentalRoute) Method() string {
	return r.local.Method()
}

// Path returns the subpath where the route responds to.
func (r *experimentalRoute) Path() string {
	return r.local.Path()
}

// Experimental will mark a route as experimental.
func Experimental(r Route) Route {
	return &experimentalRoute{
		local:   r,
		handler: experimentalHandler,
	}
}
