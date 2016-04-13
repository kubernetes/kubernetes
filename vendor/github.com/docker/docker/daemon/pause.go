package daemon

import "fmt"

// ContainerPause pauses a container
func (daemon *Daemon) ContainerPause(name string) error {
	container, err := daemon.Get(name)
	if err != nil {
		return err
	}

	if err := container.Pause(); err != nil {
		return fmt.Errorf("Cannot pause container %s: %s", name, err)
	}

	return nil
}
