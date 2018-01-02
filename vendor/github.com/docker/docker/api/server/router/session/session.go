package session

import "github.com/docker/docker/api/server/router"

// sessionRouter is a router to talk with the session controller
type sessionRouter struct {
	backend Backend
	routes  []router.Route
}

// NewRouter initializes a new session router
func NewRouter(b Backend) router.Router {
	r := &sessionRouter{
		backend: b,
	}
	r.initRoutes()
	return r
}

// Routes returns the available routers to the session controller
func (r *sessionRouter) Routes() []router.Route {
	return r.routes
}

func (r *sessionRouter) initRoutes() {
	r.routes = []router.Route{
		router.Experimental(router.NewPostRoute("/session", r.startSession)),
	}
}
