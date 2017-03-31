package containerd

import "golang.org/x/net/context"

type IO struct {
	Stdin    string
	Stdout   string
	Stderr   string
	Terminal bool
}

type CreateOpts struct {
	// Spec is the OCI runtime spec
	Spec []byte
	// Rootfs mounts to perform to gain access to the container's filesystem
	Rootfs []Mount
	// IO for the container's main process
	IO IO
}

// Runtime is responsible for the creation of containers for a certain platform,
// arch, or custom usage.
type Runtime interface {
	// Create creates a container with the provided id and options
	Create(ctx context.Context, id string, opts CreateOpts) (Container, error)
	// Containers returns all the current containers for the runtime
	Containers() ([]Container, error)
	// Delete removes the container in the runtime
	Delete(context.Context, Container) (uint32, error)
	// Events returns events for the runtime and all containers created by the runtime
	Events(context.Context) <-chan *Event
}
