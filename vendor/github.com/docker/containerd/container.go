package containerd

import "golang.org/x/net/context"

type ContainerInfo struct {
	ID      string
	Runtime string
}

type Container interface {
	// Information of the container
	Info() ContainerInfo
	// Start the container's user defined process
	Start(context.Context) error
	// State returns the container's state
	State(context.Context) (State, error)
}

type ContainerStatus int

const (
	CreatedStatus ContainerStatus = iota + 1
	RunningStatus
	StoppedStatus
	DeletedStatus
	PausedStatus
)

type State interface {
	// Status is the current status of the container
	Status() ContainerStatus
	// Pid is the main process id for the container
	Pid() uint32
}

type ContainerMonitor interface {
	Monitor(context.Context, Container) error
}
