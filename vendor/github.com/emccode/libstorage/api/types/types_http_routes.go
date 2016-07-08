package types

import (
	"github.com/akutz/gofig"
)

// Router defines an interface to specify a group of routes to add the the
// server.
type Router interface {

	// Routes returns all of the router's routes.
	Routes() []Route

	// Name returns the name of the router.
	Name() string

	// Init initializes the router.
	Init(config gofig.Config)
}

// Route defines an individual API route in the server.
type Route interface {

	// Queries add query strings that must match for a route.
	Queries(queries ...string) Route

	// Middlewares adds middleware to the route.
	Middlewares(middlewares ...Middleware) Route

	// Name returns the name of the route.
	GetName() string

	// GetHandler returns the raw function to create the http handler.
	GetHandler() APIFunc

	// GetMethod returns the http method that the route responds to.
	GetMethod() string

	// GetPath returns the subpath where the route responds to.
	GetPath() string

	// GetQueries returns the query strings for which the route should respond.
	GetQueries() []string

	// GetMiddlewares returns a list of route-specific middleware.
	GetMiddlewares() []Middleware
}
