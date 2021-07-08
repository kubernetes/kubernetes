package cow

import (
	"context"
	"io"

	"github.com/Microsoft/hcsshim/internal/schema1"
	hcsschema "github.com/Microsoft/hcsshim/internal/schema2"
)

// Process is the interface for an OS process running in a container or utility VM.
type Process interface {
	// Close releases resources associated with the process and closes the
	// writer and readers returned by Stdio. Depending on the implementation,
	// this may also terminate the process.
	Close() error
	// CloseStdin causes the process's stdin handle to receive EOF/EPIPE/whatever
	// is appropriate to indicate that no more data is available.
	CloseStdin(ctx context.Context) error
	// Pid returns the process ID.
	Pid() int
	// Stdio returns the stdio streams for a process. These may be nil if a stream
	// was not requested during CreateProcess.
	Stdio() (_ io.Writer, _ io.Reader, _ io.Reader)
	// ResizeConsole resizes the virtual terminal associated with the process.
	ResizeConsole(ctx context.Context, width, height uint16) error
	// Kill sends a SIGKILL or equivalent signal to the process and returns whether
	// the signal was delivered. It does not wait for the process to terminate.
	Kill(ctx context.Context) (bool, error)
	// Signal sends a signal to the process and returns whether the signal was
	// delivered. The input is OS specific (either
	// guestrequest.SignalProcessOptionsWCOW or
	// guestrequest.SignalProcessOptionsLCOW). It does not wait for the process
	// to terminate.
	Signal(ctx context.Context, options interface{}) (bool, error)
	// Wait waits for the process to complete, or for a connection to the process to be
	// terminated by some error condition (including calling Close).
	Wait() error
	// ExitCode returns the exit code of the process. Returns an error if the process is
	// not running.
	ExitCode() (int, error)
}

// ProcessHost is the interface for creating processes.
type ProcessHost interface {
	// CreateProcess creates a process. The configuration is host specific
	// (either hcsschema.ProcessParameters or lcow.ProcessParameters).
	CreateProcess(ctx context.Context, config interface{}) (Process, error)
	// OS returns the host's operating system, "linux" or "windows".
	OS() string
	// IsOCI specifies whether this is an OCI-compliant process host. If true,
	// then the configuration passed to CreateProcess should have an OCI process
	// spec (or nil if this is the initial process in an OCI container).
	// Otherwise, it should have the HCS-specific process parameters.
	IsOCI() bool
}

// Container is the interface for container objects, either running on the host or
// in a utility VM.
type Container interface {
	ProcessHost
	// Close releases the resources associated with the container. Depending on
	// the implementation, this may also terminate the container.
	Close() error
	// ID returns the container ID.
	ID() string
	// Properties returns the requested container properties targeting a V1 schema container.
	Properties(ctx context.Context, types ...schema1.PropertyType) (*schema1.ContainerProperties, error)
	// PropertiesV2 returns the requested container properties targeting a V2 schema container.
	PropertiesV2(ctx context.Context, types ...hcsschema.PropertyType) (*hcsschema.Properties, error)
	// Start starts a container.
	Start(ctx context.Context) error
	// Shutdown sends a shutdown request to the container (but does not wait for
	// the shutdown to complete).
	Shutdown(ctx context.Context) error
	// Terminate sends a terminate request to the container (but does not wait
	// for the terminate to complete).
	Terminate(ctx context.Context) error
	// Wait waits for the container to terminate, or for the connection to the
	// container to be terminated by some error condition (including calling
	// Close).
	Wait() error
	// Modify sends a request to modify container resources
	Modify(ctx context.Context, config interface{}) error
}
