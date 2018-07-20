package daemon

import (
	"fmt"
	"io/ioutil"
	"path/filepath"
	"runtime"
	"strings"

	containertypes "github.com/docker/docker/api/types/container"
	"github.com/docker/docker/container"
	"github.com/docker/docker/layer"
	"github.com/docker/docker/oci"
	"github.com/docker/docker/pkg/sysinfo"
	"github.com/docker/docker/pkg/system"
	"github.com/opencontainers/runtime-spec/specs-go"
	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/registry"
)

const (
	credentialSpecRegistryLocation = `SOFTWARE\Microsoft\Windows NT\CurrentVersion\Virtualization\Containers\CredentialSpecs`
	credentialSpecFileLocation     = "CredentialSpecs"
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

	if isHyperV {
		s.Windows.HyperV = &specs.WindowsHyperV{}
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
		if img.OS != runtime.GOOS {
			m.Type = "bind"
			m.Options = append(m.Options, "rbind")
			m.Options = append(m.Options, fmt.Sprintf("uvmpath=/tmp/gcs/%s/binds", c.ID))
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
		s.Process.ConsoleSize = &specs.Box{
			Height: c.HostConfig.ConsoleSize[0],
			Width:  c.HostConfig.ConsoleSize[1],
		}
	}
	s.Process.User.Username = c.Config.User

	// Get the layer path for each layer.
	max := len(img.RootFS.DiffIDs)
	for i := 1; i <= max; i++ {
		img.RootFS.DiffIDs = img.RootFS.DiffIDs[:i]
		layerPath, err := layer.GetLayerPath(daemon.stores[c.OS].layerStore, img.RootFS.ChainID())
		if err != nil {
			return nil, fmt.Errorf("failed to get layer path from graphdriver %s for ImageID %s - %s", daemon.stores[c.OS].layerStore, img.RootFS.ChainID(), err)
		}
		// Reverse order, expecting parent most first
		s.Windows.LayerFolders = append([]string{layerPath}, s.Windows.LayerFolders...)
	}
	m, err := c.RWLayer.Metadata()
	if err != nil {
		return nil, fmt.Errorf("failed to get layer metadata - %s", err)
	}
	s.Windows.LayerFolders = append(s.Windows.LayerFolders, m["dir"])

	dnsSearch := daemon.getDNSSearchSettings(c)

	// Get endpoints for the libnetwork allocated networks to the container
	var epList []string
	AllowUnqualifiedDNSQuery := false
	gwHNSID := ""
	if c.NetworkSettings != nil {
		for n := range c.NetworkSettings.Networks {
			sn, err := daemon.FindNetwork(n)
			if err != nil {
				continue
			}

			ep, err := c.GetEndpointInNetwork(sn)
			if err != nil {
				continue
			}

			data, err := ep.DriverInfo()
			if err != nil {
				continue
			}

			if data["GW_INFO"] != nil {
				gwInfo := data["GW_INFO"].(map[string]interface{})
				if gwInfo["hnsid"] != nil {
					gwHNSID = gwInfo["hnsid"].(string)
				}
			}

			if data["hnsid"] != nil {
				epList = append(epList, data["hnsid"].(string))
			}

			if data["AllowUnqualifiedDNSQuery"] != nil {
				AllowUnqualifiedDNSQuery = true
			}
		}
	}

	var networkSharedContainerID string
	if c.HostConfig.NetworkMode.IsContainer() {
		networkSharedContainerID = c.NetworkSharedContainerID
		for _, ep := range c.SharedEndpointList {
			epList = append(epList, ep)
		}
	}

	if gwHNSID != "" {
		epList = append(epList, gwHNSID)
	}

	s.Windows.Network = &specs.WindowsNetwork{
		AllowUnqualifiedDNSQuery:   AllowUnqualifiedDNSQuery,
		DNSSearchList:              dnsSearch,
		EndpointList:               epList,
		NetworkSharedContainerName: networkSharedContainerID,
	}

	if img.OS == "windows" {
		if err := daemon.createSpecWindowsFields(c, &s, isHyperV); err != nil {
			return nil, err
		}
	} else {
		// TODO @jhowardmsft LCOW Support. Modify this check when running in dual-mode
		if system.LCOWSupported() && img.OS == "linux" {
			daemon.createSpecLinuxFields(c, &s)
		}
	}

	return (*specs.Spec)(&s), nil
}

// Sets the Windows-specific fields of the OCI spec
func (daemon *Daemon) createSpecWindowsFields(c *container.Container, s *specs.Spec, isHyperV bool) error {
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
		s.Root.Path = c.BaseFS.Path() // This is not set for Hyper-V containers
		if !strings.HasSuffix(s.Root.Path, `\`) {
			s.Root.Path = s.Root.Path + `\` // Ensure a correctly formatted volume GUID path \\?\Volume{GUID}\
		}
	}

	// First boot optimization
	s.Windows.IgnoreFlushesDuringBoot = !c.HasBeenStartedBefore

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

	// Read and add credentials from the security options if a credential spec has been provided.
	if c.HostConfig.SecurityOpt != nil {
		cs := ""
		for _, sOpt := range c.HostConfig.SecurityOpt {
			sOpt = strings.ToLower(sOpt)
			if !strings.Contains(sOpt, "=") {
				return fmt.Errorf("invalid security option: no equals sign in supplied value %s", sOpt)
			}
			var splitsOpt []string
			splitsOpt = strings.SplitN(sOpt, "=", 2)
			if len(splitsOpt) != 2 {
				return fmt.Errorf("invalid security option: %s", sOpt)
			}
			if splitsOpt[0] != "credentialspec" {
				return fmt.Errorf("security option not supported: %s", splitsOpt[0])
			}

			var (
				match   bool
				csValue string
				err     error
			)
			if match, csValue = getCredentialSpec("file://", splitsOpt[1]); match {
				if csValue == "" {
					return fmt.Errorf("no value supplied for file:// credential spec security option")
				}
				if cs, err = readCredentialSpecFile(c.ID, daemon.root, filepath.Clean(csValue)); err != nil {
					return err
				}
			} else if match, csValue = getCredentialSpec("registry://", splitsOpt[1]); match {
				if csValue == "" {
					return fmt.Errorf("no value supplied for registry:// credential spec security option")
				}
				if cs, err = readCredentialSpecRegistry(c.ID, csValue); err != nil {
					return err
				}
			} else {
				return fmt.Errorf("invalid credential spec security option - value must be prefixed file:// or registry:// followed by a value")
			}
		}
		s.Windows.CredentialSpec = cs
	}

	// Assume we are not starting a container for a servicing operation
	s.Windows.Servicing = false

	return nil
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

// getCredentialSpec is a helper function to get the value of a credential spec supplied
// on the CLI, stripping the prefix
func getCredentialSpec(prefix, value string) (bool, string) {
	if strings.HasPrefix(value, prefix) {
		return true, strings.TrimPrefix(value, prefix)
	}
	return false, ""
}

// readCredentialSpecRegistry is a helper function to read a credential spec from
// the registry. If not found, we return an empty string and warn in the log.
// This allows for staging on machines which do not have the necessary components.
func readCredentialSpecRegistry(id, name string) (string, error) {
	var (
		k   registry.Key
		err error
		val string
	)
	if k, err = registry.OpenKey(registry.LOCAL_MACHINE, credentialSpecRegistryLocation, registry.QUERY_VALUE); err != nil {
		return "", fmt.Errorf("failed handling spec %q for container %s - %s could not be opened", name, id, credentialSpecRegistryLocation)
	}
	if val, _, err = k.GetStringValue(name); err != nil {
		if err == registry.ErrNotExist {
			return "", fmt.Errorf("credential spec %q for container %s as it was not found", name, id)
		}
		return "", fmt.Errorf("error %v reading credential spec %q from registry for container %s", err, name, id)
	}
	return val, nil
}

// readCredentialSpecFile is a helper function to read a credential spec from
// a file. If not found, we return an empty string and warn in the log.
// This allows for staging on machines which do not have the necessary components.
func readCredentialSpecFile(id, root, location string) (string, error) {
	if filepath.IsAbs(location) {
		return "", fmt.Errorf("invalid credential spec - file:// path cannot be absolute")
	}
	base := filepath.Join(root, credentialSpecFileLocation)
	full := filepath.Join(base, location)
	if !strings.HasPrefix(full, base) {
		return "", fmt.Errorf("invalid credential spec - file:// path must be under %s", base)
	}
	bcontents, err := ioutil.ReadFile(full)
	if err != nil {
		return "", fmt.Errorf("credential spec '%s' for container %s as the file could not be read: %q", full, id, err)
	}
	return string(bcontents[:]), nil
}
