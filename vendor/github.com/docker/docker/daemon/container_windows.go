// +build windows

package daemon

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/docker/docker/daemon/execdriver"
	"github.com/docker/docker/daemon/graphdriver/windows"
	"github.com/docker/docker/image"
	"github.com/docker/docker/pkg/archive"
	"github.com/microsoft/hcsshim"
)

// This is deliberately empty on Windows as the default path will be set by
// the container. Docker has no context of what the default path should be.
const DefaultPathEnv = ""

type Container struct {
	CommonContainer

	// Fields below here are platform specific.

	// TODO Windows. Further factoring out of unused fields will be necessary.

	// ---- START OF TEMPORARY DECLARATION ----
	// TODO Windows. Temporarily keeping fields in to assist in compilation
	// of the daemon on Windows without affecting many other files in a single
	// PR, thus making code review significantly harder. These lines will be
	// removed in subsequent PRs.

	AppArmorProfile string
	// ---- END OF TEMPORARY DECLARATION ----

}

func killProcessDirectly(container *Container) error {
	return nil
}

func (container *Container) setupContainerDns() error {
	return nil
}

func (container *Container) updateParentsHosts() error {
	return nil
}

func (container *Container) setupLinkedContainers() ([]string, error) {
	return nil, nil
}

func (container *Container) createDaemonEnvironment(linkedEnv []string) []string {
	// On Windows, nothing to link. Just return the container environment.
	return container.Config.Env
}

func (container *Container) initializeNetworking() error {
	return nil
}

func (container *Container) setupWorkingDirectory() error {
	return nil
}

func populateCommand(c *Container, env []string) error {
	en := &execdriver.Network{
		Mtu:       c.daemon.config.Mtu,
		Interface: nil,
	}

	parts := strings.SplitN(string(c.hostConfig.NetworkMode), ":", 2)
	switch parts[0] {

	case "none":
	case "default", "": // empty string to support existing containers
		if !c.Config.NetworkDisabled {
			en.Interface = &execdriver.NetworkInterface{
				MacAddress: c.Config.MacAddress,
				Bridge:     c.daemon.config.Bridge.VirtualSwitchName,
			}
		}
	default:
		return fmt.Errorf("invalid network mode: %s", c.hostConfig.NetworkMode)
	}

	pid := &execdriver.Pid{}

	// TODO Windows. This can probably be factored out.
	pid.HostPid = c.hostConfig.PidMode.IsHost()

	// TODO Windows. Resource controls to be implemented later.
	resources := &execdriver.Resources{}

	// TODO Windows. Further refactoring required (privileged/user)
	processConfig := execdriver.ProcessConfig{
		Privileged:  c.hostConfig.Privileged,
		Entrypoint:  c.Path,
		Arguments:   c.Args,
		Tty:         c.Config.Tty,
		User:        c.Config.User,
		ConsoleSize: c.hostConfig.ConsoleSize,
	}

	processConfig.Env = env

	var layerFolder string
	var layerPaths []string

	// The following is specific to the Windows driver. We do this to
	// enable VFS to continue operating for development purposes.
	if wd, ok := c.daemon.driver.(*windows.WindowsGraphDriver); ok {
		var err error
		var img *image.Image
		var ids []string

		if img, err = c.daemon.graph.Get(c.ImageID); err != nil {
			return fmt.Errorf("Failed to graph.Get on ImageID %s - %s", c.ImageID, err)
		}
		if ids, err = c.daemon.graph.ParentLayerIds(img); err != nil {
			return fmt.Errorf("Failed to get parentlayer ids %s", img.ID)
		}
		layerPaths = wd.LayerIdsToPaths(ids)
		layerFolder = filepath.Join(wd.Info().HomeDir, filepath.Base(c.ID))
	}

	// TODO Windows: Factor out remainder of unused fields.
	c.command = &execdriver.Command{
		ID:             c.ID,
		Rootfs:         c.RootfsPath(),
		ReadonlyRootfs: c.hostConfig.ReadonlyRootfs,
		InitPath:       "/.dockerinit",
		WorkingDir:     c.Config.WorkingDir,
		Network:        en,
		Pid:            pid,
		Resources:      resources,
		CapAdd:         c.hostConfig.CapAdd.Slice(),
		CapDrop:        c.hostConfig.CapDrop.Slice(),
		ProcessConfig:  processConfig,
		ProcessLabel:   c.GetProcessLabel(),
		MountLabel:     c.GetMountLabel(),
		FirstStart:     !c.HasBeenStartedBefore,
		LayerFolder:    layerFolder,
		LayerPaths:     layerPaths,
	}

	return nil
}

// GetSize, return real size, virtual size
func (container *Container) GetSize() (int64, int64) {
	// TODO Windows
	return 0, 0
}

func (container *Container) AllocateNetwork() error {
	return nil
}

func (container *Container) ExportRw() (archive.Archive, error) {
	if container.IsRunning() {
		return nil, fmt.Errorf("Cannot export a running container.")
	}
	// TODO Windows. Implementation (different to Linux)
	return nil, nil
}

func (container *Container) ReleaseNetwork() {
}

func (container *Container) RestoreNetwork() error {
	return nil
}

func disableAllActiveLinks(container *Container) {
}

func (container *Container) DisableLink(name string) {
}

func (container *Container) UnmountVolumes(forceSyscall bool) error {
	return nil
}

func (container *Container) PrepareStorage() error {
	if wd, ok := container.daemon.driver.(*windows.WindowsGraphDriver); ok {
		// Get list of paths to parent layers.
		var ids []string
		if container.ImageID != "" {
			img, err := container.daemon.graph.Get(container.ImageID)
			if err != nil {
				return err
			}

			ids, err = container.daemon.graph.ParentLayerIds(img)
			if err != nil {
				return err
			}
		}

		if err := hcsshim.PrepareLayer(wd.Info(), container.ID, wd.LayerIdsToPaths(ids)); err != nil {
			return err
		}
	}
	return nil
}

func (container *Container) CleanupStorage() error {
	if wd, ok := container.daemon.driver.(*windows.WindowsGraphDriver); ok {
		return hcsshim.UnprepareLayer(wd.Info(), container.ID)
	}
	return nil
}
