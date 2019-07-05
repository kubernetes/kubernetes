package specs

// State holds information about the runtime state of the container.
type State struct {
	// Version is the version of the specification that is supported.
	Version string `json:"ociVersion"`
	// ID is the container ID
	ID string `json:"id"`
	// Status is the runtime status of the container.
	Status string `json:"status"`
	// Pid is the process ID for the container process.
	Pid int `json:"pid,omitempty"`
	// Bundle is the path to the container's bundle directory.
	Bundle string `json:"bundle"`
	// Annotations are key values associated with the container.
	Annotations map[string]string `json:"annotations,omitempty"`
}
