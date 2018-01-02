package debug

import (
	"expvar"
	"net/http"
	"net/http/pprof"

	"github.com/docker/docker/api/server/httputils"
	"github.com/docker/docker/api/server/router"
	"golang.org/x/net/context"
)

// NewRouter creates a new debug router
// The debug router holds endpoints for debug the daemon, such as those for pprof.
func NewRouter() router.Router {
	r := &debugRouter{}
	r.initRoutes()
	return r
}

type debugRouter struct {
	routes []router.Route
}

func (r *debugRouter) initRoutes() {
	r.routes = []router.Route{
		router.NewGetRoute("/vars", frameworkAdaptHandler(expvar.Handler())),
		router.NewGetRoute("/pprof/", frameworkAdaptHandlerFunc(pprof.Index)),
		router.NewGetRoute("/pprof/cmdline", frameworkAdaptHandlerFunc(pprof.Cmdline)),
		router.NewGetRoute("/pprof/profile", frameworkAdaptHandlerFunc(pprof.Profile)),
		router.NewGetRoute("/pprof/symbol", frameworkAdaptHandlerFunc(pprof.Symbol)),
		router.NewGetRoute("/pprof/trace", frameworkAdaptHandlerFunc(pprof.Trace)),
		router.NewGetRoute("/pprof/{name}", handlePprof),
	}
}

func (r *debugRouter) Routes() []router.Route {
	return r.routes
}

func frameworkAdaptHandler(handler http.Handler) httputils.APIFunc {
	return func(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
		handler.ServeHTTP(w, r)
		return nil
	}
}

func frameworkAdaptHandlerFunc(handler http.HandlerFunc) httputils.APIFunc {
	return func(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
		handler(w, r)
		return nil
	}
}
