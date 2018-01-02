package graph

import (
	"errors"
	"sync"

	"github.com/docker/docker/daemon/graphdriver"
	"github.com/docker/docker/pkg/idtools"
)

var (
	instances map[string]graphdriver.Driver
	drivers   map[string]InitFunc
	mutex     sync.Mutex
	// ErrExist returned when driver is already registered
	ErrExist = errors.New("Driver already exists")
	// ErrNotSupported returned when operation is not supported
	ErrNotSupported = errors.New("Operation not supported")
	// ErrDriverNotFound returned when driver is not registered
	ErrDriverNotFound = errors.New("Driver implementation not found")
)

// InitFunc is the initialization function every graphdriver should implement
type InitFunc func(root string, options []string, uidMaps, gidMaps []idtools.IDMap) (graphdriver.Driver, error)

// Get returns an already registered driver
func Get(name string) (graphdriver.Driver, error) {
	if v, ok := instances[name]; ok {
		return v, nil
	}
	return nil, ErrDriverNotFound
}

// New creates a new graphdriver interface
func New(name, root string, options []string) (graphdriver.Driver, error) {
	mutex.Lock()
	defer mutex.Unlock()

	if _, ok := instances[name]; ok {
		return nil, ErrExist
	}

	if initFunc, exists := drivers[name]; exists {
		driver, err := initFunc(root, options, nil, nil)
		if err != nil {
			return nil, err
		}
		instances[name] = driver
		return driver, err
	}
	return nil, ErrNotSupported
}

// Register registers a graphdriver
func Register(name string, initFunc InitFunc) error {
	mutex.Lock()
	defer mutex.Unlock()
	if _, exists := drivers[name]; exists {
		return ErrExist
	}
	drivers[name] = initFunc
	return nil
}

func init() {
	drivers = make(map[string]InitFunc)
	instances = make(map[string]graphdriver.Driver)
}
