package container

import (
	"sync"
)

// memoryStore implements a Store in memory.
type memoryStore struct {
	s map[string]*Container
	sync.RWMutex
}

// NewMemoryStore initializes a new memory store.
func NewMemoryStore() Store {
	return &memoryStore{
		s: make(map[string]*Container),
	}
}

// Add appends a new container to the memory store.
// It overrides the id if it existed before.
func (c *memoryStore) Add(id string, cont *Container) {
	c.Lock()
	c.s[id] = cont
	c.Unlock()
}

// Get returns a container from the store by id.
func (c *memoryStore) Get(id string) *Container {
	var res *Container
	c.RLock()
	res = c.s[id]
	c.RUnlock()
	return res
}

// Delete removes a container from the store by id.
func (c *memoryStore) Delete(id string) {
	c.Lock()
	delete(c.s, id)
	c.Unlock()
}

// List returns a sorted list of containers from the store.
// The containers are ordered by creation date.
func (c *memoryStore) List() []*Container {
	containers := History(c.all())
	containers.sort()
	return containers
}

// Size returns the number of containers in the store.
func (c *memoryStore) Size() int {
	c.RLock()
	defer c.RUnlock()
	return len(c.s)
}

// First returns the first container found in the store by a given filter.
func (c *memoryStore) First(filter StoreFilter) *Container {
	for _, cont := range c.all() {
		if filter(cont) {
			return cont
		}
	}
	return nil
}

// ApplyAll calls the reducer function with every container in the store.
// This operation is asynchronous in the memory store.
// NOTE: Modifications to the store MUST NOT be done by the StoreReducer.
func (c *memoryStore) ApplyAll(apply StoreReducer) {
	wg := new(sync.WaitGroup)
	for _, cont := range c.all() {
		wg.Add(1)
		go func(container *Container) {
			apply(container)
			wg.Done()
		}(cont)
	}

	wg.Wait()
}

func (c *memoryStore) all() []*Container {
	c.RLock()
	containers := make([]*Container, 0, len(c.s))
	for _, cont := range c.s {
		containers = append(containers, cont)
	}
	c.RUnlock()
	return containers
}

var _ Store = &memoryStore{}
