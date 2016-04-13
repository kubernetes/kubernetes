package daemon

import "fmt"

// ContainerUnpause unpauses a container
func (daemon *Daemon) ContainerUnpause(name string) error {
	container, err := daemon.Get(name)
	if err != nil {
		return err
	}

	if err := container.Unpause(); err != nil {
		return fmt.Errorf("Cannot unpause container %s: %s", name, err)
	}

	return nil
}
