package swarm

import "github.com/docker/docker/api/server/router"

// swarmRouter is a router to talk with the build controller
type swarmRouter struct {
	backend Backend
	routes  []router.Route
}

// NewRouter initializes a new build router
func NewRouter(b Backend) router.Router {
	r := &swarmRouter{
		backend: b,
	}
	r.initRoutes()
	return r
}

// Routes returns the available routers to the swarm controller
func (sr *swarmRouter) Routes() []router.Route {
	return sr.routes
}

func (sr *swarmRouter) initRoutes() {
	sr.routes = []router.Route{
		router.NewPostRoute("/swarm/init", sr.initCluster),
		router.NewPostRoute("/swarm/join", sr.joinCluster),
		router.NewPostRoute("/swarm/leave", sr.leaveCluster),
		router.NewGetRoute("/swarm", sr.inspectCluster),
		router.NewGetRoute("/swarm/unlockkey", sr.getUnlockKey),
		router.NewPostRoute("/swarm/update", sr.updateCluster),
		router.NewPostRoute("/swarm/unlock", sr.unlockCluster),

		router.NewGetRoute("/services", sr.getServices),
		router.NewGetRoute("/services/{id}", sr.getService),
		router.NewPostRoute("/services/create", sr.createService),
		router.NewPostRoute("/services/{id}/update", sr.updateService),
		router.NewDeleteRoute("/services/{id}", sr.removeService),
		router.NewGetRoute("/services/{id}/logs", sr.getServiceLogs, router.WithCancel),

		router.NewGetRoute("/nodes", sr.getNodes),
		router.NewGetRoute("/nodes/{id}", sr.getNode),
		router.NewDeleteRoute("/nodes/{id}", sr.removeNode),
		router.NewPostRoute("/nodes/{id}/update", sr.updateNode),

		router.NewGetRoute("/tasks", sr.getTasks),
		router.NewGetRoute("/tasks/{id}", sr.getTask),
		router.NewGetRoute("/tasks/{id}/logs", sr.getTaskLogs, router.WithCancel),

		router.NewGetRoute("/secrets", sr.getSecrets),
		router.NewPostRoute("/secrets/create", sr.createSecret),
		router.NewDeleteRoute("/secrets/{id}", sr.removeSecret),
		router.NewGetRoute("/secrets/{id}", sr.getSecret),
		router.NewPostRoute("/secrets/{id}/update", sr.updateSecret),

		router.NewGetRoute("/configs", sr.getConfigs),
		router.NewPostRoute("/configs/create", sr.createConfig),
		router.NewDeleteRoute("/configs/{id}", sr.removeConfig),
		router.NewGetRoute("/configs/{id}", sr.getConfig),
		router.NewPostRoute("/configs/{id}/update", sr.updateConfig),
	}
}
