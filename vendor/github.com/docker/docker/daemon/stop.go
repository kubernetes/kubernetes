package daemon

import "fmt"

func (daemon *Daemon) ContainerStop(name string, seconds int) error {
	container, err := daemon.Get(name)
	if err != nil {
		return err
	}
	if !container.IsRunning() {
		return fmt.Errorf("Container already stopped")
	}
	if err := container.Stop(seconds); err != nil {
		return fmt.Errorf("Cannot stop container %s: %s\n", name, err)
	}
	return nil
}
