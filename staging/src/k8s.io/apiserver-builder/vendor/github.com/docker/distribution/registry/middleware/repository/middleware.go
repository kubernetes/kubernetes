package middleware

import (
	"fmt"

	"github.com/docker/distribution"
	"github.com/docker/distribution/context"
)

// InitFunc is the type of a RepositoryMiddleware factory function and is
// used to register the constructor for different RepositoryMiddleware backends.
type InitFunc func(ctx context.Context, repository distribution.Repository, options map[string]interface{}) (distribution.Repository, error)

var middlewares map[string]InitFunc

// Register is used to register an InitFunc for
// a RepositoryMiddleware backend with the given name.
func Register(name string, initFunc InitFunc) error {
	if middlewares == nil {
		middlewares = make(map[string]InitFunc)
	}
	if _, exists := middlewares[name]; exists {
		return fmt.Errorf("name already registered: %s", name)
	}

	middlewares[name] = initFunc

	return nil
}

// Get constructs a RepositoryMiddleware with the given options using the named backend.
func Get(ctx context.Context, name string, options map[string]interface{}, repository distribution.Repository) (distribution.Repository, error) {
	if middlewares != nil {
		if initFunc, exists := middlewares[name]; exists {
			return initFunc(ctx, repository, options)
		}
	}

	return nil, fmt.Errorf("no repository middleware registered with name: %s", name)
}
