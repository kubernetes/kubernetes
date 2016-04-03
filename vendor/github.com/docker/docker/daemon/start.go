package daemon

import (
	"fmt"
	"runtime"

	"github.com/docker/docker/runconfig"
)

func (daemon *Daemon) ContainerStart(name string, hostConfig *runconfig.HostConfig) error {
	container, err := daemon.Get(name)
	if err != nil {
		return err
	}

	if container.IsPaused() {
		return fmt.Errorf("Cannot start a paused container, try unpause instead.")
	}

	if container.IsRunning() {
		return fmt.Errorf("Container already started")
	}

	if _, err = daemon.verifyContainerSettings(hostConfig, nil); err != nil {
		return err
	}

	// Windows does not have the backwards compatibilty issue here.
	if runtime.GOOS != "windows" {
		// This is kept for backward compatibility - hostconfig should be passed when
		// creating a container, not during start.
		if hostConfig != nil {
			if err := daemon.setHostConfig(container, hostConfig); err != nil {
				return err
			}
		}
	} else {
		if hostConfig != nil {
			return fmt.Errorf("Supplying a hostconfig on start is not supported. It should be supplied on create")
		}
	}

	if err := container.Start(); err != nil {
		return fmt.Errorf("Cannot start container %s: %s", name, err)
	}

	return nil
}
