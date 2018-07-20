package daemon

import (
	"context"
	"fmt"

	"github.com/docker/docker/container"
	"github.com/sirupsen/logrus"
)

// ContainerUnpause unpauses a container
func (daemon *Daemon) ContainerUnpause(name string) error {
	container, err := daemon.GetContainer(name)
	if err != nil {
		return err
	}

	if err := daemon.containerUnpause(container); err != nil {
		return err
	}

	return nil
}

// containerUnpause resumes the container execution after the container is paused.
func (daemon *Daemon) containerUnpause(container *container.Container) error {
	container.Lock()
	defer container.Unlock()

	// We cannot unpause the container which is not paused
	if !container.Paused {
		return fmt.Errorf("Container %s is not paused", container.ID)
	}

	if err := daemon.containerd.Resume(context.Background(), container.ID); err != nil {
		return fmt.Errorf("Cannot unpause container %s: %s", container.ID, err)
	}

	container.Paused = false
	daemon.setStateCounter(container)
	daemon.updateHealthMonitor(container)
	daemon.LogContainerEvent(container, "unpause")

	if err := container.CheckpointTo(daemon.containersReplica); err != nil {
		logrus.WithError(err).Warnf("could not save container to disk")
	}

	return nil
}
