package specs

// ContainerState represents the state of a container.
type ContainerState string

const (
	// StateCreating indicates that the container is being created
	StateCreating ContainerState = "creating"

	// StateCreated indicates that the runtime has finished the create operation
	StateCreated ContainerState = "created"

	// StateRunning indicates that the container process has executed the
	// user-specified program but has not exited
	StateRunning ContainerState = "running"

	// StateStopped indicates that the container process has exited
	StateStopped ContainerState = "stopped"
)

// State holds information about the runtime state of the container.
type State struct {
	// Version is the version of the specification that is supported.
	Version string `json:"ociVersion"`
	// ID is the container ID
	ID string `json:"id"`
	// Status is the runtime status of the container.
	Status ContainerState `json:"status"`
	// Pid is the process ID for the container process.
	Pid int `json:"pid,omitempty"`
	// Bundle is the path to the container's bundle directory.
	Bundle string `json:"bundle"`
	// Annotations are key values associated with the container.
	Annotations map[string]string `json:"annotations,omitempty"`
}

const (
	// SeccompFdName is the name of the seccomp notify file descriptor.
	SeccompFdName string = "seccompFd"
)

// ContainerProcessState holds information about the state of a container process.
type ContainerProcessState struct {
	// Version is the version of the specification that is supported.
	Version string `json:"ociVersion"`
	// Fds is a string array containing the names of the file descriptors passed.
	// The index of the name in this array corresponds to index of the file
	// descriptor in the `SCM_RIGHTS` array.
	Fds []string `json:"fds"`
	// Pid is the process ID as seen by the runtime.
	Pid int `json:"pid"`
	// Opaque metadata.
	Metadata string `json:"metadata,omitempty"`
	// State of the container.
	State State `json:"state"`
}
