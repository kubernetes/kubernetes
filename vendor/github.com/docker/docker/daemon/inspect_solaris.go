package daemon

import (
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/backend"
	"github.com/docker/docker/api/types/versions/v1p19"
	"github.com/docker/docker/container"
	"github.com/docker/docker/daemon/exec"
)

// This sets platform-specific fields
func setPlatformSpecificContainerFields(container *container.Container, contJSONBase *types.ContainerJSONBase) *types.ContainerJSONBase {
	return contJSONBase
}

// containerInspectPre120 get containers for pre 1.20 APIs.
func (daemon *Daemon) containerInspectPre120(name string) (*v1p19.ContainerJSON, error) {
	return &v1p19.ContainerJSON{}, nil
}

func inspectExecProcessConfig(e *exec.Config) *backend.ExecProcessConfig {
	return &backend.ExecProcessConfig{
		Tty:        e.Tty,
		Entrypoint: e.Entrypoint,
		Arguments:  e.Args,
	}
}
