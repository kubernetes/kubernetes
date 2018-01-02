package storagemiddleware

import (
	"fmt"

	storagedriver "github.com/docker/distribution/registry/storage/driver"
)

// InitFunc is the type of a StorageMiddleware factory function and is
// used to register the constructor for different StorageMiddleware backends.
type InitFunc func(storageDriver storagedriver.StorageDriver, options map[string]interface{}) (storagedriver.StorageDriver, error)

var storageMiddlewares map[string]InitFunc

// Register is used to register an InitFunc for
// a StorageMiddleware backend with the given name.
func Register(name string, initFunc InitFunc) error {
	if storageMiddlewares == nil {
		storageMiddlewares = make(map[string]InitFunc)
	}
	if _, exists := storageMiddlewares[name]; exists {
		return fmt.Errorf("name already registered: %s", name)
	}

	storageMiddlewares[name] = initFunc

	return nil
}

// Get constructs a StorageMiddleware with the given options using the named backend.
func Get(name string, options map[string]interface{}, storageDriver storagedriver.StorageDriver) (storagedriver.StorageDriver, error) {
	if storageMiddlewares != nil {
		if initFunc, exists := storageMiddlewares[name]; exists {
			return initFunc(storageDriver, options)
		}
	}

	return nil, fmt.Errorf("no storage middleware registered with name: %s", name)
}
