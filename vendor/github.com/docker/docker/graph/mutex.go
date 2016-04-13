package graph

import "sync"

// imageMutex provides a lock per image id to protect shared resources in the
// graph. This is only used with registration but should be used when
// manipulating the layer store.
type imageMutex struct {
	mus map[string]*sync.Mutex // mutexes by image id.
	mu  sync.Mutex             // protects lock map

	// NOTE(stevvooe): The map above will grow to the size of all images ever
	// registered during a daemon run. To free these resources, we must
	// deallocate after unlock. Doing this safely is non-trivial in the face
	// of a very minor leak.
}

// Lock the provided id.
func (im *imageMutex) Lock(id string) {
	im.getImageLock(id).Lock()
}

// Unlock the provided id.
func (im *imageMutex) Unlock(id string) {
	im.getImageLock(id).Unlock()
}

// getImageLock returns the mutex for the given id. This method will never
// return nil.
func (im *imageMutex) getImageLock(id string) *sync.Mutex {
	im.mu.Lock()
	defer im.mu.Unlock()

	if im.mus == nil { // lazy
		im.mus = make(map[string]*sync.Mutex)
	}

	mu, ok := im.mus[id]
	if !ok {
		mu = new(sync.Mutex)
		im.mus[id] = mu
	}

	return mu
}
