package daemon

import (
	"fmt"
	"os"
	"path"
	"runtime"

	"github.com/Sirupsen/logrus"
)

type ContainerRmConfig struct {
	ForceRemove, RemoveVolume, RemoveLink bool
}

func (daemon *Daemon) ContainerRm(name string, config *ContainerRmConfig) error {
	container, err := daemon.Get(name)
	if err != nil {
		return err
	}

	if config.RemoveLink {
		name, err := GetFullContainerName(name)
		if err != nil {
			return err
		}
		parent, n := path.Split(name)
		if parent == "/" {
			return fmt.Errorf("Conflict, cannot remove the default name of the container")
		}
		pe := daemon.ContainerGraph().Get(parent)
		if pe == nil {
			return fmt.Errorf("Cannot get parent %s for name %s", parent, name)
		}
		parentContainer, _ := daemon.Get(pe.ID())

		if err := daemon.ContainerGraph().Delete(name); err != nil {
			return err
		}

		if parentContainer != nil {
			parentContainer.DisableLink(n)
		}

		return nil
	}

	if err := daemon.rm(container, config.ForceRemove); err != nil {
		return fmt.Errorf("Cannot destroy container %s: %v", name, err)
	}

	if config.RemoveVolume {
		container.removeMountPoints()
	}
	return nil
}

// Destroy unregisters a container from the daemon and cleanly removes its contents from the filesystem.
func (daemon *Daemon) rm(container *Container, forceRemove bool) (err error) {
	if container.IsRunning() {
		if !forceRemove {
			return fmt.Errorf("Conflict, You cannot remove a running container. Stop the container before attempting removal or use -f")
		}
		if err := container.Kill(); err != nil {
			return fmt.Errorf("Could not kill running container, cannot remove - %v", err)
		}
	}

	// stop collection of stats for the container regardless
	// if stats are currently getting collected.
	daemon.statsCollector.stopCollection(container)

	element := daemon.containers.Get(container.ID)
	if element == nil {
		return fmt.Errorf("Container %v not found - maybe it was already destroyed?", container.ID)
	}

	// Container state RemovalInProgress should be used to avoid races.
	if err = container.SetRemovalInProgress(); err != nil {
		return fmt.Errorf("Failed to set container state to RemovalInProgress: %s", err)
	}

	defer container.ResetRemovalInProgress()

	if err = container.Stop(3); err != nil {
		return err
	}

	// Mark container dead. We don't want anybody to be restarting it.
	container.SetDead()

	// Save container state to disk. So that if error happens before
	// container meta file got removed from disk, then a restart of
	// docker should not make a dead container alive.
	if err := container.ToDisk(); err != nil {
		logrus.Errorf("Error saving dying container to disk: %v", err)
	}

	// If force removal is required, delete container from various
	// indexes even if removal failed.
	defer func() {
		if err != nil && forceRemove {
			daemon.idIndex.Delete(container.ID)
			daemon.containers.Delete(container.ID)
			os.RemoveAll(container.root)
			container.LogEvent("destroy")
		}
	}()

	if _, err := daemon.containerGraph.Purge(container.ID); err != nil {
		logrus.Debugf("Unable to remove container from link graph: %s", err)
	}

	if err = daemon.driver.Remove(container.ID); err != nil {
		return fmt.Errorf("Driver %s failed to remove root filesystem %s: %s", daemon.driver, container.ID, err)
	}

	// There will not be an -init on Windows, so don't fail by not attempting to delete it
	if runtime.GOOS != "windows" {
		initID := fmt.Sprintf("%s-init", container.ID)
		if err := daemon.driver.Remove(initID); err != nil {
			return fmt.Errorf("Driver %s failed to remove init filesystem %s: %s", daemon.driver, initID, err)
		}
	}

	if err = os.RemoveAll(container.root); err != nil {
		return fmt.Errorf("Unable to remove filesystem for %v: %v", container.ID, err)
	}

	if err = daemon.execDriver.Clean(container.ID); err != nil {
		return fmt.Errorf("Unable to remove execdriver data for %s: %s", container.ID, err)
	}

	selinuxFreeLxcContexts(container.ProcessLabel)
	daemon.idIndex.Delete(container.ID)
	daemon.containers.Delete(container.ID)

	container.LogEvent("destroy")
	return nil
}

func (daemon *Daemon) DeleteVolumes(c *Container) error {
	return c.removeMountPoints()
}
