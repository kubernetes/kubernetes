package container

import (
	"github.com/docker/docker/api/server/httputils"
	"github.com/docker/docker/api/server/router"
)

type validationError struct {
	error
}

func (validationError) IsValidationError() bool {
	return true
}

// containerRouter is a router to talk with the container controller
type containerRouter struct {
	backend Backend
	decoder httputils.ContainerDecoder
	routes  []router.Route
}

// NewRouter initializes a new container router
func NewRouter(b Backend, decoder httputils.ContainerDecoder) router.Router {
	r := &containerRouter{
		backend: b,
		decoder: decoder,
	}
	r.initRoutes()
	return r
}

// Routes returns the available routes to the container controller
func (r *containerRouter) Routes() []router.Route {
	return r.routes
}

// initRoutes initializes the routes in container router
func (r *containerRouter) initRoutes() {
	r.routes = []router.Route{
		// HEAD
		router.NewHeadRoute("/containers/{name:.*}/archive", r.headContainersArchive),
		// GET
		router.NewGetRoute("/containers/json", r.getContainersJSON),
		router.NewGetRoute("/containers/{name:.*}/export", r.getContainersExport),
		router.NewGetRoute("/containers/{name:.*}/changes", r.getContainersChanges),
		router.NewGetRoute("/containers/{name:.*}/json", r.getContainersByName),
		router.NewGetRoute("/containers/{name:.*}/top", r.getContainersTop),
		router.NewGetRoute("/containers/{name:.*}/logs", r.getContainersLogs, router.WithCancel),
		router.NewGetRoute("/containers/{name:.*}/stats", r.getContainersStats, router.WithCancel),
		router.NewGetRoute("/containers/{name:.*}/attach/ws", r.wsContainersAttach),
		router.NewGetRoute("/exec/{id:.*}/json", r.getExecByID),
		router.NewGetRoute("/containers/{name:.*}/archive", r.getContainersArchive),
		// POST
		router.NewPostRoute("/containers/create", r.postContainersCreate),
		router.NewPostRoute("/containers/{name:.*}/kill", r.postContainersKill),
		router.NewPostRoute("/containers/{name:.*}/pause", r.postContainersPause),
		router.NewPostRoute("/containers/{name:.*}/unpause", r.postContainersUnpause),
		router.NewPostRoute("/containers/{name:.*}/restart", r.postContainersRestart),
		router.NewPostRoute("/containers/{name:.*}/start", r.postContainersStart),
		router.NewPostRoute("/containers/{name:.*}/stop", r.postContainersStop),
		router.NewPostRoute("/containers/{name:.*}/wait", r.postContainersWait, router.WithCancel),
		router.NewPostRoute("/containers/{name:.*}/resize", r.postContainersResize),
		router.NewPostRoute("/containers/{name:.*}/attach", r.postContainersAttach),
		router.NewPostRoute("/containers/{name:.*}/copy", r.postContainersCopy), // Deprecated since 1.8, Errors out since 1.12
		router.NewPostRoute("/containers/{name:.*}/exec", r.postContainerExecCreate),
		router.NewPostRoute("/exec/{name:.*}/start", r.postContainerExecStart),
		router.NewPostRoute("/exec/{name:.*}/resize", r.postContainerExecResize),
		router.NewPostRoute("/containers/{name:.*}/rename", r.postContainerRename),
		router.NewPostRoute("/containers/{name:.*}/update", r.postContainerUpdate),
		router.NewPostRoute("/containers/prune", r.postContainersPrune, router.WithCancel),
		// PUT
		router.NewPutRoute("/containers/{name:.*}/archive", r.putContainersArchive),
		// DELETE
		router.NewDeleteRoute("/containers/{name:.*}", r.deleteContainers),
	}
}
