package daemon

import "time"

func (daemon *Daemon) ContainerWait(name string, timeout time.Duration) (int, error) {
	container, err := daemon.Get(name)
	if err != nil {
		return -1, err
	}

	return container.WaitStop(timeout)
}
