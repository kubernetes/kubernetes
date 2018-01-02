// +build !windows

package daemon

import (
	"fmt"

	"github.com/docker/docker/container"
	"github.com/docker/docker/libcontainerd"
)

// getLibcontainerdCreateOptions callers must hold a lock on the container
func (daemon *Daemon) getLibcontainerdCreateOptions(container *container.Container) ([]libcontainerd.CreateOption, error) {
	createOptions := []libcontainerd.CreateOption{}

	// Ensure a runtime has been assigned to this container
	if container.HostConfig.Runtime == "" {
		container.HostConfig.Runtime = daemon.configStore.GetDefaultRuntimeName()
		container.CheckpointTo(daemon.containersReplica)
	}

	rt := daemon.configStore.GetRuntime(container.HostConfig.Runtime)
	if rt == nil {
		return nil, fmt.Errorf("no such runtime '%s'", container.HostConfig.Runtime)
	}
	if UsingSystemd(daemon.configStore) {
		rt.Args = append(rt.Args, "--systemd-cgroup=true")
	}
	createOptions = append(createOptions, libcontainerd.WithRuntime(rt.Path, rt.Args))

	return createOptions, nil
}
