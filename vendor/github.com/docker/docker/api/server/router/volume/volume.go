package volume

import "github.com/docker/docker/api/server/router"

// volumeRouter is a router to talk with the volumes controller
type volumeRouter struct {
	backend Backend
	routes  []router.Route
}

// NewRouter initializes a new volume router
func NewRouter(b Backend) router.Router {
	r := &volumeRouter{
		backend: b,
	}
	r.initRoutes()
	return r
}

// Routes returns the available routes to the volumes controller
func (r *volumeRouter) Routes() []router.Route {
	return r.routes
}

func (r *volumeRouter) initRoutes() {
	r.routes = []router.Route{
		// GET
		router.NewGetRoute("/volumes", r.getVolumesList),
		router.NewGetRoute("/volumes/{name:.*}", r.getVolumeByName),
		// POST
		router.NewPostRoute("/volumes/create", r.postVolumesCreate),
		router.NewPostRoute("/volumes/prune", r.postVolumesPrune, router.WithCancel),
		// DELETE
		router.NewDeleteRoute("/volumes/{name:.*}", r.deleteVolumes),
	}
}
