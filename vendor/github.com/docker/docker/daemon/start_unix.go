// +build !windows

package daemon

import (
	"fmt"
	"os/exec"
	"path/filepath"

	"github.com/containerd/containerd/linux/runctypes"
	"github.com/docker/docker/container"
	"github.com/pkg/errors"
)

func (daemon *Daemon) getRuntimeScript(container *container.Container) (string, error) {
	name := container.HostConfig.Runtime
	rt := daemon.configStore.GetRuntime(name)
	if rt == nil {
		return "", validationError{errors.Errorf("no such runtime '%s'", name)}
	}

	if len(rt.Args) > 0 {
		// First check that the target exist, as using it in a script won't
		// give us the right error
		if _, err := exec.LookPath(rt.Path); err != nil {
			return "", translateContainerdStartErr(container.Path, container.SetExitCode, err)
		}
		return filepath.Join(daemon.configStore.Root, "runtimes", name), nil
	}
	return rt.Path, nil
}

// getLibcontainerdCreateOptions callers must hold a lock on the container
func (daemon *Daemon) getLibcontainerdCreateOptions(container *container.Container) (interface{}, error) {
	// Ensure a runtime has been assigned to this container
	if container.HostConfig.Runtime == "" {
		container.HostConfig.Runtime = daemon.configStore.GetDefaultRuntimeName()
		container.CheckpointTo(daemon.containersReplica)
	}

	path, err := daemon.getRuntimeScript(container)
	if err != nil {
		return nil, err
	}
	opts := &runctypes.RuncOptions{
		Runtime: path,
		RuntimeRoot: filepath.Join(daemon.configStore.ExecRoot,
			fmt.Sprintf("runtime-%s", container.HostConfig.Runtime)),
	}

	if UsingSystemd(daemon.configStore) {
		opts.SystemdCgroup = true
	}

	return opts, nil
}
