package daemon

import (
	"context"
	"fmt"

	"github.com/docker/docker/api/types/container"
	"github.com/pkg/errors"
)

// ContainerUpdate updates configuration of the container
func (daemon *Daemon) ContainerUpdate(name string, hostConfig *container.HostConfig) (container.ContainerUpdateOKBody, error) {
	var warnings []string

	c, err := daemon.GetContainer(name)
	if err != nil {
		return container.ContainerUpdateOKBody{Warnings: warnings}, err
	}

	warnings, err = daemon.verifyContainerSettings(c.OS, hostConfig, nil, true)
	if err != nil {
		return container.ContainerUpdateOKBody{Warnings: warnings}, validationError{err}
	}

	if err := daemon.update(name, hostConfig); err != nil {
		return container.ContainerUpdateOKBody{Warnings: warnings}, err
	}

	return container.ContainerUpdateOKBody{Warnings: warnings}, nil
}

func (daemon *Daemon) update(name string, hostConfig *container.HostConfig) error {
	if hostConfig == nil {
		return nil
	}

	container, err := daemon.GetContainer(name)
	if err != nil {
		return err
	}

	restoreConfig := false
	backupHostConfig := *container.HostConfig
	defer func() {
		if restoreConfig {
			container.Lock()
			container.HostConfig = &backupHostConfig
			container.CheckpointTo(daemon.containersReplica)
			container.Unlock()
		}
	}()

	if container.RemovalInProgress || container.Dead {
		return errCannotUpdate(container.ID, fmt.Errorf("container is marked for removal and cannot be \"update\""))
	}

	container.Lock()
	if err := container.UpdateContainer(hostConfig); err != nil {
		restoreConfig = true
		container.Unlock()
		return errCannotUpdate(container.ID, err)
	}
	if err := container.CheckpointTo(daemon.containersReplica); err != nil {
		restoreConfig = true
		container.Unlock()
		return errCannotUpdate(container.ID, err)
	}
	container.Unlock()

	// if Restart Policy changed, we need to update container monitor
	if hostConfig.RestartPolicy.Name != "" {
		container.UpdateMonitor(hostConfig.RestartPolicy)
	}

	// If container is not running, update hostConfig struct is enough,
	// resources will be updated when the container is started again.
	// If container is running (including paused), we need to update configs
	// to the real world.
	if container.IsRunning() && !container.IsRestarting() {
		if err := daemon.containerd.UpdateResources(context.Background(), container.ID, toContainerdResources(hostConfig.Resources)); err != nil {
			restoreConfig = true
			// TODO: it would be nice if containerd responded with better errors here so we can classify this better.
			return errCannotUpdate(container.ID, systemError{err})
		}
	}

	daemon.LogContainerEvent(container, "update")

	return nil
}

func errCannotUpdate(containerID string, err error) error {
	return errors.Wrap(err, "Cannot update container "+containerID)
}
