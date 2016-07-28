package libcontainer

import (
	"fmt"
	"io"
	"math"
	"os"

	"github.com/opencontainers/runc/libcontainer/configs"
)

type processOperations interface {
	wait() (*os.ProcessState, error)
	signal(sig os.Signal) error
	pid() int
}

// Process specifies the configuration and IO for a process inside
// a container.
type Process struct {
	// The command to be run followed by any arguments.
	Args []string

	// Env specifies the environment variables for the process.
	Env []string

	// User will set the uid and gid of the executing process running inside the container
	// local to the container's user and group configuration.
	User string

	// Cwd will change the processes current working directory inside the container's rootfs.
	Cwd string

	// Stdin is a pointer to a reader which provides the standard input stream.
	Stdin io.Reader

	// Stdout is a pointer to a writer which receives the standard output stream.
	Stdout io.Writer

	// Stderr is a pointer to a writer which receives the standard error stream.
	Stderr io.Writer

	// ExtraFiles specifies additional open files to be inherited by the container
	ExtraFiles []*os.File

	// consolePath is the path to the console allocated to the container.
	consolePath string

	// Capabilities specify the capabilities to keep when executing the process inside the container
	// All capabilities not specified will be dropped from the processes capability mask
	Capabilities []string

	// AppArmorProfile specifies the profile to apply to the process and is
	// changed at the time the process is execed
	AppArmorProfile string

	// Label specifies the label to apply to the process.  It is commonly used by selinux
	Label string

	// NoNewPrivileges controls whether processes can gain additional privileges.
	NoNewPrivileges *bool

	// Rlimits specifies the resource limits, such as max open files, to set in the container
	// If Rlimits are not set, the container will inherit rlimits from the parent process
	Rlimits []configs.Rlimit

	ops processOperations
}

// Wait waits for the process to exit.
// Wait releases any resources associated with the Process
func (p Process) Wait() (*os.ProcessState, error) {
	if p.ops == nil {
		return nil, newGenericError(fmt.Errorf("invalid process"), NoProcessOps)
	}
	return p.ops.wait()
}

// Pid returns the process ID
func (p Process) Pid() (int, error) {
	// math.MinInt32 is returned here, because it's invalid value
	// for the kill() system call.
	if p.ops == nil {
		return math.MinInt32, newGenericError(fmt.Errorf("invalid process"), NoProcessOps)
	}
	return p.ops.pid(), nil
}

// Signal sends a signal to the Process.
func (p Process) Signal(sig os.Signal) error {
	if p.ops == nil {
		return newGenericError(fmt.Errorf("invalid process"), NoProcessOps)
	}
	return p.ops.signal(sig)
}

// IO holds the process's STDIO
type IO struct {
	Stdin  io.WriteCloser
	Stdout io.ReadCloser
	Stderr io.ReadCloser
}

// NewConsole creates new console for process and returns it
func (p *Process) NewConsole(rootuid int) (Console, error) {
	console, err := NewConsole(rootuid, rootuid)
	if err != nil {
		return nil, err
	}
	p.consolePath = console.Path()
	return console, nil
}

// ConsoleFromPath sets the process's console with the path provided
func (p *Process) ConsoleFromPath(path string) error {
	if p.consolePath != "" {
		return newGenericError(fmt.Errorf("console path already exists for process"), ConsoleExists)
	}
	p.consolePath = path
	return nil
}
