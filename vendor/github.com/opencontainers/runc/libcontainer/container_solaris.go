package libcontainer

// State represents a running container's state
type State struct {
	BaseState

	// Platform specific fields below here
}

// A libcontainer container object.
//
// Each container is thread-safe within the same process. Since a container can
// be destroyed by a separate process, any function may return that the container
// was not found.
type Container interface {
	BaseContainer

	// Methods below here are platform specific

}
