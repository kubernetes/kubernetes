// +build linux freebsd

package daemon

import (
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/docker/docker/container"
	"github.com/docker/docker/daemon/links"
	"github.com/docker/docker/pkg/idtools"
	"github.com/docker/docker/pkg/mount"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/runconfig"
	"github.com/docker/libnetwork"
	"github.com/opencontainers/selinux/go-selinux/label"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

func (daemon *Daemon) setupLinkedContainers(container *container.Container) ([]string, error) {
	var env []string
	children := daemon.children(container)

	bridgeSettings := container.NetworkSettings.Networks[runconfig.DefaultDaemonNetworkMode().NetworkName()]
	if bridgeSettings == nil || bridgeSettings.EndpointSettings == nil {
		return nil, nil
	}

	for linkAlias, child := range children {
		if !child.IsRunning() {
			return nil, fmt.Errorf("Cannot link to a non running container: %s AS %s", child.Name, linkAlias)
		}

		childBridgeSettings := child.NetworkSettings.Networks[runconfig.DefaultDaemonNetworkMode().NetworkName()]
		if childBridgeSettings == nil || childBridgeSettings.EndpointSettings == nil {
			return nil, fmt.Errorf("container %s not attached to default bridge network", child.ID)
		}

		link := links.NewLink(
			bridgeSettings.IPAddress,
			childBridgeSettings.IPAddress,
			linkAlias,
			child.Config.Env,
			child.Config.ExposedPorts,
		)

		env = append(env, link.ToEnv()...)
	}

	return env, nil
}

func (daemon *Daemon) getIpcContainer(container *container.Container) (*container.Container, error) {
	containerID := container.HostConfig.IpcMode.Container()
	container, err := daemon.GetContainer(containerID)
	if err != nil {
		return nil, errors.Wrapf(err, "cannot join IPC of a non running container: %s", container.ID)
	}
	return container, daemon.checkContainer(container, containerIsRunning, containerIsNotRestarting)
}

func (daemon *Daemon) getPidContainer(container *container.Container) (*container.Container, error) {
	containerID := container.HostConfig.PidMode.Container()
	container, err := daemon.GetContainer(containerID)
	if err != nil {
		return nil, errors.Wrapf(err, "cannot join PID of a non running container: %s", container.ID)
	}
	return container, daemon.checkContainer(container, containerIsRunning, containerIsNotRestarting)
}

func containerIsRunning(c *container.Container) error {
	if !c.IsRunning() {
		return errors.Errorf("container %s is not running", c.ID)
	}
	return nil
}

func containerIsNotRestarting(c *container.Container) error {
	if c.IsRestarting() {
		return errContainerIsRestarting(c.ID)
	}
	return nil
}

func (daemon *Daemon) setupIpcDirs(c *container.Container) error {
	var err error

	c.ShmPath, err = c.ShmResourcePath()
	if err != nil {
		return err
	}

	if c.HostConfig.IpcMode.IsContainer() {
		ic, err := daemon.getIpcContainer(c)
		if err != nil {
			return err
		}
		c.ShmPath = ic.ShmPath
	} else if c.HostConfig.IpcMode.IsHost() {
		if _, err := os.Stat("/dev/shm"); err != nil {
			return fmt.Errorf("/dev/shm is not mounted, but must be for --ipc=host")
		}
		c.ShmPath = "/dev/shm"
	} else {
		rootIDs := daemon.idMappings.RootPair()
		if !c.HasMountFor("/dev/shm") {
			shmPath, err := c.ShmResourcePath()
			if err != nil {
				return err
			}

			if err := idtools.MkdirAllAndChown(shmPath, 0700, rootIDs); err != nil {
				return err
			}

			shmSize := int64(daemon.configStore.ShmSize)
			if c.HostConfig.ShmSize != 0 {
				shmSize = c.HostConfig.ShmSize
			}
			shmproperty := "mode=1777,size=" + strconv.FormatInt(shmSize, 10)
			if err := unix.Mount("shm", shmPath, "tmpfs", uintptr(unix.MS_NOEXEC|unix.MS_NOSUID|unix.MS_NODEV), label.FormatMountLabel(shmproperty, c.GetMountLabel())); err != nil {
				return fmt.Errorf("mounting shm tmpfs: %s", err)
			}
			if err := os.Chown(shmPath, rootIDs.UID, rootIDs.GID); err != nil {
				return err
			}
		}

	}

	return nil
}

func (daemon *Daemon) setupSecretDir(c *container.Container) (setupErr error) {
	if len(c.SecretReferences) == 0 {
		return nil
	}

	localMountPath := c.SecretMountPath()
	logrus.Debugf("secrets: setting up secret dir: %s", localMountPath)

	// retrieve possible remapped range start for root UID, GID
	rootIDs := daemon.idMappings.RootPair()
	// create tmpfs
	if err := idtools.MkdirAllAndChown(localMountPath, 0700, rootIDs); err != nil {
		return errors.Wrap(err, "error creating secret local mount path")
	}

	defer func() {
		if setupErr != nil {
			// cleanup
			_ = detachMounted(localMountPath)

			if err := os.RemoveAll(localMountPath); err != nil {
				logrus.Errorf("error cleaning up secret mount: %s", err)
			}
		}
	}()

	tmpfsOwnership := fmt.Sprintf("uid=%d,gid=%d", rootIDs.UID, rootIDs.GID)
	if err := mount.Mount("tmpfs", localMountPath, "tmpfs", "nodev,nosuid,noexec,"+tmpfsOwnership); err != nil {
		return errors.Wrap(err, "unable to setup secret mount")
	}

	if c.DependencyStore == nil {
		return fmt.Errorf("secret store is not initialized")
	}

	for _, s := range c.SecretReferences {
		// TODO (ehazlett): use type switch when more are supported
		if s.File == nil {
			logrus.Error("secret target type is not a file target")
			continue
		}

		// secrets are created in the SecretMountPath on the host, at a
		// single level
		fPath := c.SecretFilePath(*s)
		if err := idtools.MkdirAllAndChown(filepath.Dir(fPath), 0700, rootIDs); err != nil {
			return errors.Wrap(err, "error creating secret mount path")
		}

		logrus.WithFields(logrus.Fields{
			"name": s.File.Name,
			"path": fPath,
		}).Debug("injecting secret")
		secret, err := c.DependencyStore.Secrets().Get(s.SecretID)
		if err != nil {
			return errors.Wrap(err, "unable to get secret from secret store")
		}
		if err := ioutil.WriteFile(fPath, secret.Spec.Data, s.File.Mode); err != nil {
			return errors.Wrap(err, "error injecting secret")
		}

		uid, err := strconv.Atoi(s.File.UID)
		if err != nil {
			return err
		}
		gid, err := strconv.Atoi(s.File.GID)
		if err != nil {
			return err
		}

		if err := os.Chown(fPath, rootIDs.UID+uid, rootIDs.GID+gid); err != nil {
			return errors.Wrap(err, "error setting ownership for secret")
		}
	}

	label.Relabel(localMountPath, c.MountLabel, false)

	// remount secrets ro
	if err := mount.Mount("tmpfs", localMountPath, "tmpfs", "remount,ro,"+tmpfsOwnership); err != nil {
		return errors.Wrap(err, "unable to remount secret dir as readonly")
	}

	return nil
}

func (daemon *Daemon) setupConfigDir(c *container.Container) (setupErr error) {
	if len(c.ConfigReferences) == 0 {
		return nil
	}

	localPath := c.ConfigsDirPath()
	logrus.Debugf("configs: setting up config dir: %s", localPath)

	// retrieve possible remapped range start for root UID, GID
	rootIDs := daemon.idMappings.RootPair()
	// create tmpfs
	if err := idtools.MkdirAllAndChown(localPath, 0700, rootIDs); err != nil {
		return errors.Wrap(err, "error creating config dir")
	}

	defer func() {
		if setupErr != nil {
			if err := os.RemoveAll(localPath); err != nil {
				logrus.Errorf("error cleaning up config dir: %s", err)
			}
		}
	}()

	if c.DependencyStore == nil {
		return fmt.Errorf("config store is not initialized")
	}

	for _, configRef := range c.ConfigReferences {
		// TODO (ehazlett): use type switch when more are supported
		if configRef.File == nil {
			logrus.Error("config target type is not a file target")
			continue
		}

		fPath := c.ConfigFilePath(*configRef)

		log := logrus.WithFields(logrus.Fields{"name": configRef.File.Name, "path": fPath})

		if err := idtools.MkdirAllAndChown(filepath.Dir(fPath), 0700, rootIDs); err != nil {
			return errors.Wrap(err, "error creating config path")
		}

		log.Debug("injecting config")
		config, err := c.DependencyStore.Configs().Get(configRef.ConfigID)
		if err != nil {
			return errors.Wrap(err, "unable to get config from config store")
		}
		if err := ioutil.WriteFile(fPath, config.Spec.Data, configRef.File.Mode); err != nil {
			return errors.Wrap(err, "error injecting config")
		}

		uid, err := strconv.Atoi(configRef.File.UID)
		if err != nil {
			return err
		}
		gid, err := strconv.Atoi(configRef.File.GID)
		if err != nil {
			return err
		}

		if err := os.Chown(fPath, rootIDs.UID+uid, rootIDs.GID+gid); err != nil {
			return errors.Wrap(err, "error setting ownership for config")
		}
	}

	return nil
}

func killProcessDirectly(cntr *container.Container) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Block until the container to stops or timeout.
	status := <-cntr.Wait(ctx, container.WaitConditionNotRunning)
	if status.Err() != nil {
		// Ensure that we don't kill ourselves
		if pid := cntr.GetPID(); pid != 0 {
			logrus.Infof("Container %s failed to exit within 10 seconds of kill - trying direct SIGKILL", stringid.TruncateID(cntr.ID))
			if err := unix.Kill(pid, 9); err != nil {
				if err != unix.ESRCH {
					return err
				}
				e := errNoSuchProcess{pid, 9}
				logrus.Debug(e)
				return e
			}
		}
	}
	return nil
}

func detachMounted(path string) error {
	return unix.Unmount(path, unix.MNT_DETACH)
}

func isLinkable(child *container.Container) bool {
	// A container is linkable only if it belongs to the default network
	_, ok := child.NetworkSettings.Networks[runconfig.DefaultDaemonNetworkMode().NetworkName()]
	return ok
}

func enableIPOnPredefinedNetwork() bool {
	return false
}

func (daemon *Daemon) isNetworkHotPluggable() bool {
	return true
}

func setupPathsAndSandboxOptions(container *container.Container, sboxOptions *[]libnetwork.SandboxOption) error {
	var err error

	container.HostsPath, err = container.GetRootResourcePath("hosts")
	if err != nil {
		return err
	}
	*sboxOptions = append(*sboxOptions, libnetwork.OptionHostsPath(container.HostsPath))

	container.ResolvConfPath, err = container.GetRootResourcePath("resolv.conf")
	if err != nil {
		return err
	}
	*sboxOptions = append(*sboxOptions, libnetwork.OptionResolvConfPath(container.ResolvConfPath))
	return nil
}

func (daemon *Daemon) initializeNetworkingPaths(container *container.Container, nc *container.Container) error {
	container.HostnamePath = nc.HostnamePath
	container.HostsPath = nc.HostsPath
	container.ResolvConfPath = nc.ResolvConfPath
	return nil
}
