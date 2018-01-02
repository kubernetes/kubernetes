package daemon

import (
	"fmt"
	"net/http"
	"runtime"
	"strings"
	"syscall"
	"time"

	"google.golang.org/grpc"

	apierrors "github.com/docker/docker/api/errors"
	"github.com/docker/docker/api/types"
	containertypes "github.com/docker/docker/api/types/container"
	"github.com/docker/docker/container"
	"github.com/sirupsen/logrus"
)

// ContainerStart starts a container.
func (daemon *Daemon) ContainerStart(name string, hostConfig *containertypes.HostConfig, checkpoint string, checkpointDir string) error {
	if checkpoint != "" && !daemon.HasExperimental() {
		return apierrors.NewBadRequestError(fmt.Errorf("checkpoint is only supported in experimental mode"))
	}

	container, err := daemon.GetContainer(name)
	if err != nil {
		return err
	}

	if container.IsPaused() {
		return fmt.Errorf("Cannot start a paused container, try unpause instead.")
	}

	if container.IsRunning() {
		err := fmt.Errorf("Container already started")
		return apierrors.NewErrorWithStatusCode(err, http.StatusNotModified)
	}

	// Windows does not have the backwards compatibility issue here.
	if runtime.GOOS != "windows" {
		// This is kept for backward compatibility - hostconfig should be passed when
		// creating a container, not during start.
		if hostConfig != nil {
			logrus.Warn("DEPRECATED: Setting host configuration options when the container starts is deprecated and has been removed in Docker 1.12")
			oldNetworkMode := container.HostConfig.NetworkMode
			if err := daemon.setSecurityOptions(container, hostConfig); err != nil {
				return err
			}
			if err := daemon.mergeAndVerifyLogConfig(&hostConfig.LogConfig); err != nil {
				return err
			}
			if err := daemon.setHostConfig(container, hostConfig); err != nil {
				return err
			}
			newNetworkMode := container.HostConfig.NetworkMode
			if string(oldNetworkMode) != string(newNetworkMode) {
				// if user has change the network mode on starting, clean up the
				// old networks. It is a deprecated feature and has been removed in Docker 1.12
				container.NetworkSettings.Networks = nil
				if err := container.CheckpointTo(daemon.containersReplica); err != nil {
					return err
				}
			}
			container.InitDNSHostConfig()
		}
	} else {
		if hostConfig != nil {
			return fmt.Errorf("Supplying a hostconfig on start is not supported. It should be supplied on create")
		}
	}

	// check if hostConfig is in line with the current system settings.
	// It may happen cgroups are umounted or the like.
	if _, err = daemon.verifyContainerSettings(container.HostConfig, nil, false); err != nil {
		return err
	}
	// Adapt for old containers in case we have updates in this function and
	// old containers never have chance to call the new function in create stage.
	if hostConfig != nil {
		if err := daemon.adaptContainerSettings(container.HostConfig, false); err != nil {
			return err
		}
	}

	return daemon.containerStart(container, checkpoint, checkpointDir, true)
}

// containerStart prepares the container to run by setting up everything the
// container needs, such as storage and networking, as well as links
// between containers. The container is left waiting for a signal to
// begin running.
func (daemon *Daemon) containerStart(container *container.Container, checkpoint string, checkpointDir string, resetRestartManager bool) (err error) {
	start := time.Now()
	container.Lock()
	defer container.Unlock()

	if resetRestartManager && container.Running { // skip this check if already in restarting step and resetRestartManager==false
		return nil
	}

	if container.RemovalInProgress || container.Dead {
		return fmt.Errorf("Container is marked for removal and cannot be started.")
	}

	// if we encounter an error during start we need to ensure that any other
	// setup has been cleaned up properly
	defer func() {
		if err != nil {
			container.SetError(err)
			// if no one else has set it, make sure we don't leave it at zero
			if container.ExitCode() == 0 {
				container.SetExitCode(128)
			}
			if err := container.CheckpointTo(daemon.containersReplica); err != nil {
				logrus.Errorf("%s: failed saving state on start failure: %v", container.ID, err)
			}
			container.Reset(false)

			daemon.Cleanup(container)
			// if containers AutoRemove flag is set, remove it after clean up
			if container.HostConfig.AutoRemove {
				container.Unlock()
				if err := daemon.ContainerRm(container.ID, &types.ContainerRmConfig{ForceRemove: true, RemoveVolume: true}); err != nil {
					logrus.Errorf("can't remove container %s: %v", container.ID, err)
				}
				container.Lock()
			}
		}
	}()

	if err := daemon.conditionalMountOnStart(container); err != nil {
		return err
	}

	if err := daemon.initializeNetworking(container); err != nil {
		return err
	}

	spec, err := daemon.createSpec(container)
	if err != nil {
		return err
	}

	createOptions, err := daemon.getLibcontainerdCreateOptions(container)
	if err != nil {
		return err
	}

	if resetRestartManager {
		container.ResetRestartManager(true)
	}

	if checkpointDir == "" {
		checkpointDir = container.CheckpointDir()
	}

	if daemon.saveApparmorConfig(container); err != nil {
		return err
	}

	if err := daemon.containerd.Create(container.ID, checkpoint, checkpointDir, *spec, container.InitializeStdio, createOptions...); err != nil {
		errDesc := grpc.ErrorDesc(err)
		contains := func(s1, s2 string) bool {
			return strings.Contains(strings.ToLower(s1), s2)
		}
		logrus.Errorf("Create container failed with error: %s", errDesc)
		// if we receive an internal error from the initial start of a container then lets
		// return it instead of entering the restart loop
		// set to 127 for container cmd not found/does not exist)
		if contains(errDesc, container.Path) &&
			(contains(errDesc, "executable file not found") ||
				contains(errDesc, "no such file or directory") ||
				contains(errDesc, "system cannot find the file specified")) {
			container.SetExitCode(127)
		}
		// set to 126 for container cmd can't be invoked errors
		if contains(errDesc, syscall.EACCES.Error()) {
			container.SetExitCode(126)
		}

		// attempted to mount a file onto a directory, or a directory onto a file, maybe from user specified bind mounts
		if contains(errDesc, syscall.ENOTDIR.Error()) {
			errDesc += ": Are you trying to mount a directory onto a file (or vice-versa)? Check if the specified host path exists and is the expected type"
			container.SetExitCode(127)
		}

		return fmt.Errorf("%s", errDesc)
	}

	containerActions.WithValues("start").UpdateSince(start)

	return nil
}

// Cleanup releases any network resources allocated to the container along with any rules
// around how containers are linked together.  It also unmounts the container's root filesystem.
func (daemon *Daemon) Cleanup(container *container.Container) {
	daemon.releaseNetwork(container)

	container.UnmountIpcMounts(detachMounted)

	if err := daemon.conditionalUnmountOnCleanup(container); err != nil {
		// FIXME: remove once reference counting for graphdrivers has been refactored
		// Ensure that all the mounts are gone
		if mountid, err := daemon.stores[container.Platform].layerStore.GetMountID(container.ID); err == nil {
			daemon.cleanupMountsByID(mountid)
		}
	}

	if err := container.UnmountSecrets(); err != nil {
		logrus.Warnf("%s cleanup: failed to unmount secrets: %s", container.ID, err)
	}

	for _, eConfig := range container.ExecCommands.Commands() {
		daemon.unregisterExecCommand(container, eConfig)
	}

	if container.BaseFS != "" {
		if err := container.UnmountVolumes(daemon.LogVolumeEvent); err != nil {
			logrus.Warnf("%s cleanup: Failed to umount volumes: %v", container.ID, err)
		}
	}
	container.CancelAttachContext()
}
