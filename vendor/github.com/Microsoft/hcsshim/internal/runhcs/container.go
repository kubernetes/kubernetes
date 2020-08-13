package runhcs

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"syscall"
	"time"

	"github.com/Microsoft/go-winio/pkg/guid"
)

// ContainerState represents the platform agnostic pieces relating to a
// running container's status and state
type ContainerState struct {
	// Version is the OCI version for the container
	Version string `json:"ociVersion"`
	// ID is the container ID
	ID string `json:"id"`
	// InitProcessPid is the init process id in the parent namespace
	InitProcessPid int `json:"pid"`
	// Status is the current status of the container, running, paused, ...
	Status string `json:"status"`
	// Bundle is the path on the filesystem to the bundle
	Bundle string `json:"bundle"`
	// Rootfs is a path to a directory containing the container's root filesystem.
	Rootfs string `json:"rootfs"`
	// Created is the unix timestamp for the creation time of the container in UTC
	Created time.Time `json:"created"`
	// Annotations is the user defined annotations added to the config.
	Annotations map[string]string `json:"annotations,omitempty"`
	// The owner of the state directory (the owner of the container).
	Owner string `json:"owner"`
}

// GetErrorFromPipe returns reads from `pipe` and verifies if the operation
// returned success or error. If error converts that to an error and returns. If
// `p` is not nill will issue a `Kill` and `Wait` for exit.
func GetErrorFromPipe(pipe io.Reader, p *os.Process) error {
	serr, err := ioutil.ReadAll(pipe)
	if err != nil {
		return err
	}

	if bytes.Equal(serr, ShimSuccess) {
		return nil
	}

	extra := ""
	if p != nil {
		p.Kill()
		state, err := p.Wait()
		if err != nil {
			panic(err)
		}
		extra = fmt.Sprintf(", exit code %d", state.Sys().(syscall.WaitStatus).ExitCode)
	}
	if len(serr) == 0 {
		return fmt.Errorf("unknown shim failure%s", extra)
	}

	return errors.New(string(serr))
}

// VMPipePath returns the named pipe path for the vm shim.
func VMPipePath(hostUniqueID guid.GUID) string {
	return SafePipePath("runhcs-vm-" + hostUniqueID.String())
}
