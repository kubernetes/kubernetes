package daemon

import (
	"fmt"

	"github.com/docker/docker/api/errors"
)

func (d *Daemon) imageNotExistToErrcode(err error) error {
	if dne, isDNE := err.(ErrImageDoesNotExist); isDNE {
		return errors.NewRequestNotFoundError(dne)
	}
	return err
}

type errNotRunning struct {
	containerID string
}

func (e errNotRunning) Error() string {
	return fmt.Sprintf("Container %s is not running", e.containerID)
}

func (e errNotRunning) ContainerIsRunning() bool {
	return false
}

func errContainerIsRestarting(containerID string) error {
	err := fmt.Errorf("Container %s is restarting, wait until the container is running", containerID)
	return errors.NewRequestConflictError(err)
}

func errExecNotFound(id string) error {
	err := fmt.Errorf("No such exec instance '%s' found in daemon", id)
	return errors.NewRequestNotFoundError(err)
}

func errExecPaused(id string) error {
	err := fmt.Errorf("Container %s is paused, unpause the container before exec", id)
	return errors.NewRequestConflictError(err)
}

type errNotFound struct {
	containerID string
}

func (e errNotFound) Error() string {
	return fmt.Sprintf("Container %s is not found", e.containerID)
}

func (e errNotFound) ContainerNotFound() bool {
	return true
}
