package containerd

import "errors"

var (
	ErrUnknownRuntime    = errors.New("unknown runtime")
	ErrContainerExists   = errors.New("container with id already exists")
	ErrContainerNotExist = errors.New("container does not exist")
	ErrRuntimeNotExist   = errors.New("runtime does not exist")
)
