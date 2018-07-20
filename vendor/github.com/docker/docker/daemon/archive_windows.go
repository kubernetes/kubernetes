package daemon

import (
	"errors"

	containertypes "github.com/docker/docker/api/types/container"
	"github.com/docker/docker/container"
)

// checkIfPathIsInAVolume checks if the path is in a volume. If it is, it
// cannot be in a read-only volume. If it  is not in a volume, the container
// cannot be configured with a read-only rootfs.
//
// This is a no-op on Windows which does not support read-only volumes, or
// extracting to a mount point inside a volume. TODO Windows: FIXME Post-TP5
func checkIfPathIsInAVolume(container *container.Container, absPath string) (bool, error) {
	return false, nil
}

// isOnlineFSOperationPermitted returns an error if an online filesystem operation
// is not permitted (such as stat or for copying). Running Hyper-V containers
// cannot have their file-system interrogated from the host as the filter is
// loaded inside the utility VM, not the host.
// IMPORTANT: The container lock must NOT be held when calling this function.
func (daemon *Daemon) isOnlineFSOperationPermitted(container *container.Container) error {
	if !container.IsRunning() {
		return nil
	}

	// Determine isolation. If not specified in the hostconfig, use daemon default.
	actualIsolation := container.HostConfig.Isolation
	if containertypes.Isolation.IsDefault(containertypes.Isolation(actualIsolation)) {
		actualIsolation = daemon.defaultIsolation
	}
	if containertypes.Isolation.IsHyperV(actualIsolation) {
		return errors.New("filesystem operations against a running Hyper-V container are not supported")
	}
	return nil
}
