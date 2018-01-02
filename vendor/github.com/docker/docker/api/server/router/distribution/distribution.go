package distribution

import "github.com/docker/docker/api/server/router"

// distributionRouter is a router to talk with the registry
type distributionRouter struct {
	backend Backend
	routes  []router.Route
}

// NewRouter initializes a new distribution router
func NewRouter(backend Backend) router.Router {
	r := &distributionRouter{
		backend: backend,
	}
	r.initRoutes()
	return r
}

// Routes returns the available routes
func (r *distributionRouter) Routes() []router.Route {
	return r.routes
}

// initRoutes initializes the routes in the distribution router
func (r *distributionRouter) initRoutes() {
	r.routes = []router.Route{
		// GET
		router.NewGetRoute("/distribution/{name:.*}/json", r.getDistributionInfo),
	}
}
