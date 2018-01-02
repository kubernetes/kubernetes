package daemon

import (
	containertypes "github.com/docker/docker/api/types/container"
	"github.com/docker/docker/container"
	"github.com/docker/docker/oci"
	"github.com/docker/docker/pkg/sysinfo"
	"github.com/docker/docker/pkg/system"
	"github.com/opencontainers/runtime-spec/specs-go"
	"golang.org/x/sys/windows"
)

func (daemon *Daemon) createSpec(c *container.Container) (*specs.Spec, error) {
	img, err := daemon.GetImage(string(c.ImageID))
	if err != nil {
		return nil, err
	}

	s := oci.DefaultOSSpec(img.OS)

	linkedEnv, err := daemon.setupLinkedContainers(c)
	if err != nil {
		return nil, err
	}

	// Note, unlike Unix, we do NOT call into SetupWorkingDirectory as
	// this is done in VMCompute. Further, we couldn't do it for Hyper-V
	// containers anyway.

	// In base spec
	s.Hostname = c.FullHostname()

	if err := daemon.setupSecretDir(c); err != nil {
		return nil, err
	}

	if err := daemon.setupConfigDir(c); err != nil {
		return nil, err
	}

	// In s.Mounts
	mounts, err := daemon.setupMounts(c)
	if err != nil {
		return nil, err
	}

	var isHyperV bool
	if c.HostConfig.Isolation.IsDefault() {
		// Container using default isolation, so take the default from the daemon configuration
		isHyperV = daemon.defaultIsolation.IsHyperV()
	} else {
		// Container may be requesting an explicit isolation mode.
		isHyperV = c.HostConfig.Isolation.IsHyperV()
	}

	// If the container has not been started, and has configs or secrets
	// secrets, create symlinks to each config and secret. If it has been
	// started before, the symlinks should have already been created. Also, it
	// is important to not mount a Hyper-V  container that has been started
	// before, to protect the host from the container; for example, from
	// malicious mutation of NTFS data structures.
	if !c.HasBeenStartedBefore && (len(c.SecretReferences) > 0 || len(c.ConfigReferences) > 0) {
		// The container file system is mounted before this function is called,
		// except for Hyper-V containers, so mount it here in that case.
		if isHyperV {
			if err := daemon.Mount(c); err != nil {
				return nil, err
			}
			defer daemon.Unmount(c)
		}
		if err := c.CreateSecretSymlinks(); err != nil {
			return nil, err
		}
		if err := c.CreateConfigSymlinks(); err != nil {
			return nil, err
		}
	}

	if m := c.SecretMounts(); m != nil {
		mounts = append(mounts, m...)
	}

	if m := c.ConfigMounts(); m != nil {
		mounts = append(mounts, m...)
	}

	for _, mount := range mounts {
		m := specs.Mount{
			Source:      mount.Source,
			Destination: mount.Destination,
		}
		if !mount.Writable {
			m.Options = append(m.Options, "ro")
		}
		s.Mounts = append(s.Mounts, m)
	}

	// In s.Process
	s.Process.Args = append([]string{c.Path}, c.Args...)
	if !c.Config.ArgsEscaped && img.OS == "windows" {
		s.Process.Args = escapeArgs(s.Process.Args)
	}

	s.Process.Cwd = c.Config.WorkingDir
	s.Process.Env = c.CreateDaemonEnvironment(c.Config.Tty, linkedEnv)
	if c.Config.Tty {
		s.Process.Terminal = c.Config.Tty
		s.Process.ConsoleSize.Height = c.HostConfig.ConsoleSize[0]
		s.Process.ConsoleSize.Width = c.HostConfig.ConsoleSize[1]
	}
	s.Process.User.Username = c.Config.User

	if img.OS == "windows" {
		daemon.createSpecWindowsFields(c, &s, isHyperV)
	} else {
		// TODO @jhowardmsft LCOW Support. Modify this check when running in dual-mode
		if system.LCOWSupported() && img.OS == "linux" {
			daemon.createSpecLinuxFields(c, &s)
		}
	}

	return (*specs.Spec)(&s), nil
}

// Sets the Windows-specific fields of the OCI spec
func (daemon *Daemon) createSpecWindowsFields(c *container.Container, s *specs.Spec, isHyperV bool) {
	if len(s.Process.Cwd) == 0 {
		// We default to C:\ to workaround the oddity of the case that the
		// default directory for cmd running as LocalSystem (or
		// ContainerAdministrator) is c:\windows\system32. Hence docker run
		// <image> cmd will by default end in c:\windows\system32, rather
		// than 'root' (/) on Linux. The oddity is that if you have a dockerfile
		// which has no WORKDIR and has a COPY file ., . will be interpreted
		// as c:\. Hence, setting it to default of c:\ makes for consistency.
		s.Process.Cwd = `C:\`
	}

	s.Root.Readonly = false // Windows does not support a read-only root filesystem
	if !isHyperV {
		s.Root.Path = c.BaseFS // This is not set for Hyper-V containers
	}

	// In s.Windows.Resources
	cpuShares := uint16(c.HostConfig.CPUShares)
	cpuMaximum := uint16(c.HostConfig.CPUPercent) * 100
	cpuCount := uint64(c.HostConfig.CPUCount)
	if c.HostConfig.NanoCPUs > 0 {
		if isHyperV {
			cpuCount = uint64(c.HostConfig.NanoCPUs / 1e9)
			leftoverNanoCPUs := c.HostConfig.NanoCPUs % 1e9
			if leftoverNanoCPUs != 0 {
				cpuCount++
				cpuMaximum = uint16(c.HostConfig.NanoCPUs / int64(cpuCount) / (1e9 / 10000))
				if cpuMaximum < 1 {
					// The requested NanoCPUs is so small that we rounded to 0, use 1 instead
					cpuMaximum = 1
				}
			}
		} else {
			cpuMaximum = uint16(c.HostConfig.NanoCPUs / int64(sysinfo.NumCPU()) / (1e9 / 10000))
			if cpuMaximum < 1 {
				// The requested NanoCPUs is so small that we rounded to 0, use 1 instead
				cpuMaximum = 1
			}
		}
	}
	memoryLimit := uint64(c.HostConfig.Memory)
	s.Windows.Resources = &specs.WindowsResources{
		CPU: &specs.WindowsCPUResources{
			Maximum: &cpuMaximum,
			Shares:  &cpuShares,
			Count:   &cpuCount,
		},
		Memory: &specs.WindowsMemoryResources{
			Limit: &memoryLimit,
		},
		Storage: &specs.WindowsStorageResources{
			Bps:  &c.HostConfig.IOMaximumBandwidth,
			Iops: &c.HostConfig.IOMaximumIOps,
		},
	}
}

// Sets the Linux-specific fields of the OCI spec
// TODO: @jhowardmsft LCOW Support. We need to do a lot more pulling in what can
// be pulled in from oci_linux.go.
func (daemon *Daemon) createSpecLinuxFields(c *container.Container, s *specs.Spec) {
	if len(s.Process.Cwd) == 0 {
		s.Process.Cwd = `/`
	}
	s.Root.Path = "rootfs"
	s.Root.Readonly = c.HostConfig.ReadonlyRootfs
}

func escapeArgs(args []string) []string {
	escapedArgs := make([]string, len(args))
	for i, a := range args {
		escapedArgs[i] = windows.EscapeArg(a)
	}
	return escapedArgs
}

// mergeUlimits merge the Ulimits from HostConfig with daemon defaults, and update HostConfig
// It will do nothing on non-Linux platform
func (daemon *Daemon) mergeUlimits(c *containertypes.HostConfig) {
	return
}
