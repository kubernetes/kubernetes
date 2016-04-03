package daemon

func (daemon *Daemon) ContainerResize(name string, height, width int) error {
	container, err := daemon.Get(name)
	if err != nil {
		return err
	}

	return container.Resize(height, width)
}

func (daemon *Daemon) ContainerExecResize(name string, height, width int) error {
	execConfig, err := daemon.getExecConfig(name)
	if err != nil {
		return err
	}

	return execConfig.Resize(height, width)
}
