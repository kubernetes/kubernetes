package libcontainerd

import (
	containerd "github.com/containerd/containerd/api/grpc/types"
	"github.com/opencontainers/runtime-spec/specs-go"
)

// Process contains information to start a specific application inside the container.
type Process struct {
	// Terminal creates an interactive terminal for the container.
	Terminal bool `json:"terminal"`
	// User specifies user information for the process.
	User *specs.User `json:"user"`
	// Args specifies the binary and arguments for the application to execute.
	Args []string `json:"args"`
	// Env populates the process environment for the process.
	Env []string `json:"env,omitempty"`
	// Cwd is the current working directory for the process and must be
	// relative to the container's root.
	Cwd *string `json:"cwd"`
	// Capabilities are linux capabilities that are kept for the container.
	Capabilities []string `json:"capabilities,omitempty"`
	// Rlimits specifies rlimit options to apply to the process.
	Rlimits []specs.LinuxRlimit `json:"rlimits,omitempty"`
	// ApparmorProfile specifies the apparmor profile for the container.
	ApparmorProfile *string `json:"apparmorProfile,omitempty"`
	// SelinuxLabel specifies the selinux context that the container process is run as.
	SelinuxLabel *string `json:"selinuxLabel,omitempty"`
}

// StateInfo contains description about the new state container has entered.
type StateInfo struct {
	CommonStateInfo

	// Platform specific StateInfo
	OOMKilled bool
}

// Stats contains a stats properties from containerd.
type Stats containerd.StatsResponse

// Summary contains a container summary from containerd
type Summary struct{}

// Resources defines updatable container resource values.
type Resources containerd.UpdateResource

// Checkpoints contains the details of a checkpoint
type Checkpoints containerd.ListCheckpointResponse
