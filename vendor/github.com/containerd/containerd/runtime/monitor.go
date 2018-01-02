package runtime

// TaskMonitor provides an interface for monitoring of containers within containerd
type TaskMonitor interface {
	// Monitor adds the provided container to the monitor
	Monitor(Task) error
	// Stop stops and removes the provided container from the monitor
	Stop(Task) error
}

// NewMultiTaskMonitor returns a new TaskMonitor broadcasting to the provided monitors
func NewMultiTaskMonitor(monitors ...TaskMonitor) TaskMonitor {
	return &multiTaskMonitor{
		monitors: monitors,
	}
}

// NewNoopMonitor is a task monitor that does nothing
func NewNoopMonitor() TaskMonitor {
	return &noopTaskMonitor{}
}

type noopTaskMonitor struct {
}

func (mm *noopTaskMonitor) Monitor(c Task) error {
	return nil
}

func (mm *noopTaskMonitor) Stop(c Task) error {
	return nil
}

type multiTaskMonitor struct {
	monitors []TaskMonitor
}

func (mm *multiTaskMonitor) Monitor(c Task) error {
	for _, m := range mm.monitors {
		if err := m.Monitor(c); err != nil {
			return err
		}
	}
	return nil
}

func (mm *multiTaskMonitor) Stop(c Task) error {
	for _, m := range mm.monitors {
		if err := m.Stop(c); err != nil {
			return err
		}
	}
	return nil
}
