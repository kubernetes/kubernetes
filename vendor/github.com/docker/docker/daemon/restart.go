package daemon

import "fmt"

func (daemon *Daemon) ContainerRestart(name string, seconds int) error {
	container, err := daemon.Get(name)
	if err != nil {
		return err
	}
	if err := container.Restart(seconds); err != nil {
		return fmt.Errorf("Cannot restart container %s: %s\n", name, err)
	}
	return nil
}
