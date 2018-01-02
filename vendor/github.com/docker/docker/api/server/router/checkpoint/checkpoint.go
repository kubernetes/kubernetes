package checkpoint

import (
	"github.com/docker/docker/api/server/httputils"
	"github.com/docker/docker/api/server/router"
)

// checkpointRouter is a router to talk with the checkpoint controller
type checkpointRouter struct {
	backend Backend
	decoder httputils.ContainerDecoder
	routes  []router.Route
}

// NewRouter initializes a new checkpoint router
func NewRouter(b Backend, decoder httputils.ContainerDecoder) router.Router {
	r := &checkpointRouter{
		backend: b,
		decoder: decoder,
	}
	r.initRoutes()
	return r
}

// Routes returns the available routers to the checkpoint controller
func (r *checkpointRouter) Routes() []router.Route {
	return r.routes
}

func (r *checkpointRouter) initRoutes() {
	r.routes = []router.Route{
		router.NewGetRoute("/containers/{name:.*}/checkpoints", r.getContainerCheckpoints, router.Experimental),
		router.NewPostRoute("/containers/{name:.*}/checkpoints", r.postContainerCheckpoint, router.Experimental),
		router.NewDeleteRoute("/containers/{name}/checkpoints/{checkpoint}", r.deleteContainerCheckpoint, router.Experimental),
	}
}
