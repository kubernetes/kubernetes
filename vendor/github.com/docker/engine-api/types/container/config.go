package container

import (
	"time"

	"github.com/docker/engine-api/types/strslice"
	"github.com/docker/go-connections/nat"
)

// HealthConfig holds configuration settings for the HEALTHCHECK feature.
type HealthConfig struct {
	// Test is the test to perform to check that the container is healthy.
	// An empty slice means to inherit the default.
	// The options are:
	// {} : inherit healthcheck
	// {"NONE"} : disable healthcheck
	// {"CMD", args...} : exec arguments directly
	// {"CMD-SHELL", command} : run command with system's default shell
	Test []string `json:",omitempty"`

	// Zero means to inherit. Durations are expressed as integer nanoseconds.
	Interval time.Duration `json:",omitempty"` // Interval is the time to wait between checks.
	Timeout  time.Duration `json:",omitempty"` // Timeout is the time to wait before considering the check to have hung.

	// Retries is the number of consecutive failures needed to consider a container as unhealthy.
	// Zero means inherit.
	Retries int `json:",omitempty"`
}

// Config contains the configuration data about a container.
// It should hold only portable information about the container.
// Here, "portable" means "independent from the host we are running on".
// Non-portable information *should* appear in HostConfig.
// All fields added to this struct must be marked `omitempty` to keep getting
// predictable hashes from the old `v1Compatibility` configuration.
type Config struct {
	Hostname        string                // Hostname
	Domainname      string                // Domainname
	User            string                // User that will run the command(s) inside the container
	AttachStdin     bool                  // Attach the standard input, makes possible user interaction
	AttachStdout    bool                  // Attach the standard output
	AttachStderr    bool                  // Attach the standard error
	ExposedPorts    map[nat.Port]struct{} `json:",omitempty"` // List of exposed ports
	Tty             bool                  // Attach standard streams to a tty, including stdin if it is not closed.
	OpenStdin       bool                  // Open stdin
	StdinOnce       bool                  // If true, close stdin after the 1 attached client disconnects.
	Env             []string              // List of environment variable to set in the container
	Cmd             strslice.StrSlice     // Command to run when starting the container
	Healthcheck     *HealthConfig         `json:",omitempty"` // Healthcheck describes how to check the container is healthy
	ArgsEscaped     bool                  `json:",omitempty"` // True if command is already escaped (Windows specific)
	Image           string                // Name of the image as it was passed by the operator (eg. could be symbolic)
	Volumes         map[string]struct{}   // List of volumes (mounts) used for the container
	WorkingDir      string                // Current directory (PWD) in the command will be launched
	Entrypoint      strslice.StrSlice     // Entrypoint to run when starting the container
	NetworkDisabled bool                  `json:",omitempty"` // Is network disabled
	MacAddress      string                `json:",omitempty"` // Mac Address of the container
	OnBuild         []string              // ONBUILD metadata that were defined on the image Dockerfile
	Labels          map[string]string     // List of labels set to this container
	StopSignal      string                `json:",omitempty"` // Signal to stop a container
	StopTimeout     *int                  `json:",omitempty"` // Timeout (in seconds) to stop a container
	Shell           strslice.StrSlice     `json:",omitempty"` // Shell for shell-form of RUN, CMD, ENTRYPOINT
}
