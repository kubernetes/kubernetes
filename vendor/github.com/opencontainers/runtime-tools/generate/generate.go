// Package generate implements functions generating container config files.
package generate

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	rspec "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/opencontainers/runtime-tools/generate/seccomp"
	"github.com/opencontainers/runtime-tools/validate"
	"github.com/syndtr/gocapability/capability"
)

var (
	// Namespaces include the names of supported namespaces.
	Namespaces = []string{"network", "pid", "mount", "ipc", "uts", "user", "cgroup"}

	// we don't care about order...and this is way faster...
	removeFunc = func(s []string, i int) []string {
		s[i] = s[len(s)-1]
		return s[:len(s)-1]
	}
)

// Generator represents a generator for a container config.
type Generator struct {
	Config       *rspec.Spec
	HostSpecific bool
	// This is used to keep a cache of the ENVs added to improve
	// performance when adding a huge number of ENV variables
	envMap map[string]int
}

// ExportOptions have toggles for exporting only certain parts of the specification
type ExportOptions struct {
	Seccomp bool // seccomp toggles if only seccomp should be exported
}

// New creates a configuration Generator with the default
// configuration for the target operating system.
func New(os string) (generator Generator, err error) {
	if os != "linux" && os != "solaris" && os != "windows" {
		return generator, fmt.Errorf("no defaults configured for %s", os)
	}

	config := rspec.Spec{
		Version:  rspec.Version,
		Hostname: "mrsdalloway",
	}

	if os == "windows" {
		config.Process = &rspec.Process{
			Args: []string{
				"cmd",
			},
			Cwd: `C:\`,
		}
		config.Windows = &rspec.Windows{}
	} else {
		config.Root = &rspec.Root{
			Path:     "rootfs",
			Readonly: false,
		}
		config.Process = &rspec.Process{
			Terminal: false,
			Args: []string{
				"sh",
			},
		}
	}

	if os == "linux" || os == "solaris" {
		config.Process.User = rspec.User{}
		config.Process.Env = []string{
			"PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
			"TERM=xterm",
		}
		config.Process.Cwd = "/"
		config.Process.Rlimits = []rspec.POSIXRlimit{
			{
				Type: "RLIMIT_NOFILE",
				Hard: uint64(1024),
				Soft: uint64(1024),
			},
		}
	}

	if os == "linux" {
		config.Process.Capabilities = &rspec.LinuxCapabilities{
			Bounding: []string{
				"CAP_CHOWN",
				"CAP_DAC_OVERRIDE",
				"CAP_FSETID",
				"CAP_FOWNER",
				"CAP_MKNOD",
				"CAP_NET_RAW",
				"CAP_SETGID",
				"CAP_SETUID",
				"CAP_SETFCAP",
				"CAP_SETPCAP",
				"CAP_NET_BIND_SERVICE",
				"CAP_SYS_CHROOT",
				"CAP_KILL",
				"CAP_AUDIT_WRITE",
			},
			Permitted: []string{
				"CAP_CHOWN",
				"CAP_DAC_OVERRIDE",
				"CAP_FSETID",
				"CAP_FOWNER",
				"CAP_MKNOD",
				"CAP_NET_RAW",
				"CAP_SETGID",
				"CAP_SETUID",
				"CAP_SETFCAP",
				"CAP_SETPCAP",
				"CAP_NET_BIND_SERVICE",
				"CAP_SYS_CHROOT",
				"CAP_KILL",
				"CAP_AUDIT_WRITE",
			},
			Inheritable: []string{
				"CAP_CHOWN",
				"CAP_DAC_OVERRIDE",
				"CAP_FSETID",
				"CAP_FOWNER",
				"CAP_MKNOD",
				"CAP_NET_RAW",
				"CAP_SETGID",
				"CAP_SETUID",
				"CAP_SETFCAP",
				"CAP_SETPCAP",
				"CAP_NET_BIND_SERVICE",
				"CAP_SYS_CHROOT",
				"CAP_KILL",
				"CAP_AUDIT_WRITE",
			},
			Effective: []string{
				"CAP_CHOWN",
				"CAP_DAC_OVERRIDE",
				"CAP_FSETID",
				"CAP_FOWNER",
				"CAP_MKNOD",
				"CAP_NET_RAW",
				"CAP_SETGID",
				"CAP_SETUID",
				"CAP_SETFCAP",
				"CAP_SETPCAP",
				"CAP_NET_BIND_SERVICE",
				"CAP_SYS_CHROOT",
				"CAP_KILL",
				"CAP_AUDIT_WRITE",
			},
			Ambient: []string{
				"CAP_CHOWN",
				"CAP_DAC_OVERRIDE",
				"CAP_FSETID",
				"CAP_FOWNER",
				"CAP_MKNOD",
				"CAP_NET_RAW",
				"CAP_SETGID",
				"CAP_SETUID",
				"CAP_SETFCAP",
				"CAP_SETPCAP",
				"CAP_NET_BIND_SERVICE",
				"CAP_SYS_CHROOT",
				"CAP_KILL",
				"CAP_AUDIT_WRITE",
			},
		}
		config.Mounts = []rspec.Mount{
			{
				Destination: "/proc",
				Type:        "proc",
				Source:      "proc",
				Options:     []string{"nosuid", "noexec", "nodev"},
			},
			{
				Destination: "/dev",
				Type:        "tmpfs",
				Source:      "tmpfs",
				Options:     []string{"nosuid", "strictatime", "mode=755", "size=65536k"},
			},
			{
				Destination: "/dev/pts",
				Type:        "devpts",
				Source:      "devpts",
				Options:     []string{"nosuid", "noexec", "newinstance", "ptmxmode=0666", "mode=0620", "gid=5"},
			},
			{
				Destination: "/dev/shm",
				Type:        "tmpfs",
				Source:      "shm",
				Options:     []string{"nosuid", "noexec", "nodev", "mode=1777", "size=65536k"},
			},
			{
				Destination: "/dev/mqueue",
				Type:        "mqueue",
				Source:      "mqueue",
				Options:     []string{"nosuid", "noexec", "nodev"},
			},
			{
				Destination: "/sys",
				Type:        "sysfs",
				Source:      "sysfs",
				Options:     []string{"nosuid", "noexec", "nodev", "ro"},
			},
		}
		config.Linux = &rspec.Linux{
			Resources: &rspec.LinuxResources{
				Devices: []rspec.LinuxDeviceCgroup{
					{
						Allow:  false,
						Access: "rwm",
					},
				},
			},
			Namespaces: []rspec.LinuxNamespace{
				{
					Type: "pid",
				},
				{
					Type: "network",
				},
				{
					Type: "ipc",
				},
				{
					Type: "uts",
				},
				{
					Type: "mount",
				},
			},
			Seccomp: seccomp.DefaultProfile(&config),
		}
	}

	envCache := map[string]int{}
	if config.Process != nil {
		envCache = createEnvCacheMap(config.Process.Env)
	}

	return Generator{Config: &config, envMap: envCache}, nil
}

// NewFromSpec creates a configuration Generator from a given
// configuration.
//
// Deprecated: Replace with:
//
//   generator := Generator{Config: config}
func NewFromSpec(config *rspec.Spec) Generator {
	envCache := map[string]int{}
	if config != nil && config.Process != nil {
		envCache = createEnvCacheMap(config.Process.Env)
	}

	return Generator{
		Config: config,
		envMap: envCache,
	}
}

// NewFromFile loads the template specified in a file into a
// configuration Generator.
func NewFromFile(path string) (Generator, error) {
	cf, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return Generator{}, fmt.Errorf("template configuration at %s not found", path)
		}
		return Generator{}, err
	}
	defer cf.Close()

	return NewFromTemplate(cf)
}

// NewFromTemplate loads the template from io.Reader into a
// configuration Generator.
func NewFromTemplate(r io.Reader) (Generator, error) {
	var config rspec.Spec
	if err := json.NewDecoder(r).Decode(&config); err != nil {
		return Generator{}, err
	}

	envCache := map[string]int{}
	if config.Process != nil {
		envCache = createEnvCacheMap(config.Process.Env)
	}

	return Generator{
		Config: &config,
		envMap: envCache,
	}, nil
}

// createEnvCacheMap creates a hash map with the ENV variables given by the config
func createEnvCacheMap(env []string) map[string]int {
	envMap := make(map[string]int, len(env))
	for i, val := range env {
		envMap[val] = i
	}
	return envMap
}

// SetSpec sets the configuration in the Generator g.
//
// Deprecated: Replace with:
//
//   Use generator.Config = config
func (g *Generator) SetSpec(config *rspec.Spec) {
	g.Config = config
}

// Spec gets the configuration from the Generator g.
//
// Deprecated: Replace with generator.Config.
func (g *Generator) Spec() *rspec.Spec {
	return g.Config
}

// Save writes the configuration into w.
func (g *Generator) Save(w io.Writer, exportOpts ExportOptions) (err error) {
	var data []byte

	if g.Config.Linux != nil {
		buf, err := json.Marshal(g.Config.Linux)
		if err != nil {
			return err
		}
		if string(buf) == "{}" {
			g.Config.Linux = nil
		}
	}

	if exportOpts.Seccomp {
		data, err = json.MarshalIndent(g.Config.Linux.Seccomp, "", "\t")
	} else {
		data, err = json.MarshalIndent(g.Config, "", "\t")
	}
	if err != nil {
		return err
	}

	_, err = w.Write(data)
	if err != nil {
		return err
	}

	return nil
}

// SaveToFile writes the configuration into a file.
func (g *Generator) SaveToFile(path string, exportOpts ExportOptions) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return g.Save(f, exportOpts)
}

// SetVersion sets g.Config.Version.
func (g *Generator) SetVersion(version string) {
	g.initConfig()
	g.Config.Version = version
}

// SetRootPath sets g.Config.Root.Path.
func (g *Generator) SetRootPath(path string) {
	g.initConfigRoot()
	g.Config.Root.Path = path
}

// SetRootReadonly sets g.Config.Root.Readonly.
func (g *Generator) SetRootReadonly(b bool) {
	g.initConfigRoot()
	g.Config.Root.Readonly = b
}

// SetHostname sets g.Config.Hostname.
func (g *Generator) SetHostname(s string) {
	g.initConfig()
	g.Config.Hostname = s
}

// SetOCIVersion sets g.Config.Version.
func (g *Generator) SetOCIVersion(s string) {
	g.initConfig()
	g.Config.Version = s
}

// ClearAnnotations clears g.Config.Annotations.
func (g *Generator) ClearAnnotations() {
	if g.Config == nil {
		return
	}
	g.Config.Annotations = make(map[string]string)
}

// AddAnnotation adds an annotation into g.Config.Annotations.
func (g *Generator) AddAnnotation(key, value string) {
	g.initConfigAnnotations()
	g.Config.Annotations[key] = value
}

// RemoveAnnotation remove an annotation from g.Config.Annotations.
func (g *Generator) RemoveAnnotation(key string) {
	if g.Config == nil || g.Config.Annotations == nil {
		return
	}
	delete(g.Config.Annotations, key)
}

// RemoveHostname removes g.Config.Hostname, setting it to an empty string.
func (g *Generator) RemoveHostname() {
	if g.Config == nil {
		return
	}
	g.Config.Hostname = ""
}

// SetProcessConsoleSize sets g.Config.Process.ConsoleSize.
func (g *Generator) SetProcessConsoleSize(width, height uint) {
	g.initConfigProcessConsoleSize()
	g.Config.Process.ConsoleSize.Width = width
	g.Config.Process.ConsoleSize.Height = height
}

// SetProcessUID sets g.Config.Process.User.UID.
func (g *Generator) SetProcessUID(uid uint32) {
	g.initConfigProcess()
	g.Config.Process.User.UID = uid
}

// SetProcessUsername sets g.Config.Process.User.Username.
func (g *Generator) SetProcessUsername(username string) {
	g.initConfigProcess()
	g.Config.Process.User.Username = username
}

// SetProcessGID sets g.Config.Process.User.GID.
func (g *Generator) SetProcessGID(gid uint32) {
	g.initConfigProcess()
	g.Config.Process.User.GID = gid
}

// SetProcessCwd sets g.Config.Process.Cwd.
func (g *Generator) SetProcessCwd(cwd string) {
	g.initConfigProcess()
	g.Config.Process.Cwd = cwd
}

// SetProcessNoNewPrivileges sets g.Config.Process.NoNewPrivileges.
func (g *Generator) SetProcessNoNewPrivileges(b bool) {
	g.initConfigProcess()
	g.Config.Process.NoNewPrivileges = b
}

// SetProcessTerminal sets g.Config.Process.Terminal.
func (g *Generator) SetProcessTerminal(b bool) {
	g.initConfigProcess()
	g.Config.Process.Terminal = b
}

// SetProcessApparmorProfile sets g.Config.Process.ApparmorProfile.
func (g *Generator) SetProcessApparmorProfile(prof string) {
	g.initConfigProcess()
	g.Config.Process.ApparmorProfile = prof
}

// SetProcessArgs sets g.Config.Process.Args.
func (g *Generator) SetProcessArgs(args []string) {
	g.initConfigProcess()
	g.Config.Process.Args = args
}

// ClearProcessEnv clears g.Config.Process.Env.
func (g *Generator) ClearProcessEnv() {
	if g.Config == nil || g.Config.Process == nil {
		return
	}
	g.Config.Process.Env = []string{}
	// Clear out the env cache map as well
	g.envMap = map[string]int{}
}

// AddProcessEnv adds name=value into g.Config.Process.Env, or replaces an
// existing entry with the given name.
func (g *Generator) AddProcessEnv(name, value string) {
	if name == "" {
		return
	}

	g.initConfigProcess()
	g.addEnv(fmt.Sprintf("%s=%s", name, value), name)
}

// AddMultipleProcessEnv adds multiple name=value into g.Config.Process.Env, or replaces
// existing entries with the given name.
func (g *Generator) AddMultipleProcessEnv(envs []string) {
	g.initConfigProcess()

	for _, val := range envs {
		split := strings.SplitN(val, "=", 2)
		g.addEnv(val, split[0])
	}
}

// addEnv looks through adds ENV to the Process and checks envMap for
// any duplicates
// This is called by both AddMultipleProcessEnv and AddProcessEnv
func (g *Generator) addEnv(env, key string) {
	if idx, ok := g.envMap[key]; ok {
		// The ENV exists in the cache, so change its value in g.Config.Process.Env
		g.Config.Process.Env[idx] = env
	} else {
		// else the env doesn't exist, so add it and add it's index to g.envMap
		g.Config.Process.Env = append(g.Config.Process.Env, env)
		g.envMap[key] = len(g.Config.Process.Env) - 1
	}
}

// AddProcessRlimits adds rlimit into g.Config.Process.Rlimits.
func (g *Generator) AddProcessRlimits(rType string, rHard uint64, rSoft uint64) {
	g.initConfigProcess()
	for i, rlimit := range g.Config.Process.Rlimits {
		if rlimit.Type == rType {
			g.Config.Process.Rlimits[i].Hard = rHard
			g.Config.Process.Rlimits[i].Soft = rSoft
			return
		}
	}

	newRlimit := rspec.POSIXRlimit{
		Type: rType,
		Hard: rHard,
		Soft: rSoft,
	}
	g.Config.Process.Rlimits = append(g.Config.Process.Rlimits, newRlimit)
}

// RemoveProcessRlimits removes a rlimit from g.Config.Process.Rlimits.
func (g *Generator) RemoveProcessRlimits(rType string) {
	if g.Config == nil || g.Config.Process == nil {
		return
	}
	for i, rlimit := range g.Config.Process.Rlimits {
		if rlimit.Type == rType {
			g.Config.Process.Rlimits = append(g.Config.Process.Rlimits[:i], g.Config.Process.Rlimits[i+1:]...)
			return
		}
	}
}

// ClearProcessRlimits clear g.Config.Process.Rlimits.
func (g *Generator) ClearProcessRlimits() {
	if g.Config == nil || g.Config.Process == nil {
		return
	}
	g.Config.Process.Rlimits = []rspec.POSIXRlimit{}
}

// ClearProcessAdditionalGids clear g.Config.Process.AdditionalGids.
func (g *Generator) ClearProcessAdditionalGids() {
	if g.Config == nil || g.Config.Process == nil {
		return
	}
	g.Config.Process.User.AdditionalGids = []uint32{}
}

// AddProcessAdditionalGid adds an additional gid into g.Config.Process.AdditionalGids.
func (g *Generator) AddProcessAdditionalGid(gid uint32) {
	g.initConfigProcess()
	for _, group := range g.Config.Process.User.AdditionalGids {
		if group == gid {
			return
		}
	}
	g.Config.Process.User.AdditionalGids = append(g.Config.Process.User.AdditionalGids, gid)
}

// SetProcessSelinuxLabel sets g.Config.Process.SelinuxLabel.
func (g *Generator) SetProcessSelinuxLabel(label string) {
	g.initConfigProcess()
	g.Config.Process.SelinuxLabel = label
}

// SetLinuxCgroupsPath sets g.Config.Linux.CgroupsPath.
func (g *Generator) SetLinuxCgroupsPath(path string) {
	g.initConfigLinux()
	g.Config.Linux.CgroupsPath = path
}

// SetLinuxIntelRdtL3CacheSchema sets g.Config.Linux.IntelRdt.L3CacheSchema
func (g *Generator) SetLinuxIntelRdtL3CacheSchema(schema string) {
	g.initConfigLinuxIntelRdt()
	g.Config.Linux.IntelRdt.L3CacheSchema = schema
}

// SetLinuxMountLabel sets g.Config.Linux.MountLabel.
func (g *Generator) SetLinuxMountLabel(label string) {
	g.initConfigLinux()
	g.Config.Linux.MountLabel = label
}

// SetProcessOOMScoreAdj sets g.Config.Process.OOMScoreAdj.
func (g *Generator) SetProcessOOMScoreAdj(adj int) {
	g.initConfigProcess()
	g.Config.Process.OOMScoreAdj = &adj
}

// SetLinuxResourcesBlockIOLeafWeight sets g.Config.Linux.Resources.BlockIO.LeafWeight.
func (g *Generator) SetLinuxResourcesBlockIOLeafWeight(weight uint16) {
	g.initConfigLinuxResourcesBlockIO()
	g.Config.Linux.Resources.BlockIO.LeafWeight = &weight
}

// AddLinuxResourcesBlockIOLeafWeightDevice adds or sets g.Config.Linux.Resources.BlockIO.WeightDevice.LeafWeight.
func (g *Generator) AddLinuxResourcesBlockIOLeafWeightDevice(major int64, minor int64, weight uint16) {
	g.initConfigLinuxResourcesBlockIO()
	for i, weightDevice := range g.Config.Linux.Resources.BlockIO.WeightDevice {
		if weightDevice.Major == major && weightDevice.Minor == minor {
			g.Config.Linux.Resources.BlockIO.WeightDevice[i].LeafWeight = &weight
			return
		}
	}
	weightDevice := new(rspec.LinuxWeightDevice)
	weightDevice.Major = major
	weightDevice.Minor = minor
	weightDevice.LeafWeight = &weight
	g.Config.Linux.Resources.BlockIO.WeightDevice = append(g.Config.Linux.Resources.BlockIO.WeightDevice, *weightDevice)
}

// DropLinuxResourcesBlockIOLeafWeightDevice drops a item form g.Config.Linux.Resources.BlockIO.WeightDevice.LeafWeight
func (g *Generator) DropLinuxResourcesBlockIOLeafWeightDevice(major int64, minor int64) {
	if g.Config == nil || g.Config.Linux == nil || g.Config.Linux.Resources == nil || g.Config.Linux.Resources.BlockIO == nil {
		return
	}

	for i, weightDevice := range g.Config.Linux.Resources.BlockIO.WeightDevice {
		if weightDevice.Major == major && weightDevice.Minor == minor {
			if weightDevice.Weight != nil {
				newWeightDevice := new(rspec.LinuxWeightDevice)
				newWeightDevice.Major = major
				newWeightDevice.Minor = minor
				newWeightDevice.Weight = weightDevice.Weight
				g.Config.Linux.Resources.BlockIO.WeightDevice[i] = *newWeightDevice
			} else {
				g.Config.Linux.Resources.BlockIO.WeightDevice = append(g.Config.Linux.Resources.BlockIO.WeightDevice[:i], g.Config.Linux.Resources.BlockIO.WeightDevice[i+1:]...)
			}
			return
		}
	}
}

// SetLinuxResourcesBlockIOWeight sets g.Config.Linux.Resources.BlockIO.Weight.
func (g *Generator) SetLinuxResourcesBlockIOWeight(weight uint16) {
	g.initConfigLinuxResourcesBlockIO()
	g.Config.Linux.Resources.BlockIO.Weight = &weight
}

// AddLinuxResourcesBlockIOWeightDevice adds or sets g.Config.Linux.Resources.BlockIO.WeightDevice.Weight.
func (g *Generator) AddLinuxResourcesBlockIOWeightDevice(major int64, minor int64, weight uint16) {
	g.initConfigLinuxResourcesBlockIO()
	for i, weightDevice := range g.Config.Linux.Resources.BlockIO.WeightDevice {
		if weightDevice.Major == major && weightDevice.Minor == minor {
			g.Config.Linux.Resources.BlockIO.WeightDevice[i].Weight = &weight
			return
		}
	}
	weightDevice := new(rspec.LinuxWeightDevice)
	weightDevice.Major = major
	weightDevice.Minor = minor
	weightDevice.Weight = &weight
	g.Config.Linux.Resources.BlockIO.WeightDevice = append(g.Config.Linux.Resources.BlockIO.WeightDevice, *weightDevice)
}

// DropLinuxResourcesBlockIOWeightDevice drops a item form g.Config.Linux.Resources.BlockIO.WeightDevice.Weight
func (g *Generator) DropLinuxResourcesBlockIOWeightDevice(major int64, minor int64) {
	if g.Config == nil || g.Config.Linux == nil || g.Config.Linux.Resources == nil || g.Config.Linux.Resources.BlockIO == nil {
		return
	}

	for i, weightDevice := range g.Config.Linux.Resources.BlockIO.WeightDevice {
		if weightDevice.Major == major && weightDevice.Minor == minor {
			if weightDevice.LeafWeight != nil {
				newWeightDevice := new(rspec.LinuxWeightDevice)
				newWeightDevice.Major = major
				newWeightDevice.Minor = minor
				newWeightDevice.LeafWeight = weightDevice.LeafWeight
				g.Config.Linux.Resources.BlockIO.WeightDevice[i] = *newWeightDevice
			} else {
				g.Config.Linux.Resources.BlockIO.WeightDevice = append(g.Config.Linux.Resources.BlockIO.WeightDevice[:i], g.Config.Linux.Resources.BlockIO.WeightDevice[i+1:]...)
			}
			return
		}
	}
}

// AddLinuxResourcesBlockIOThrottleReadBpsDevice adds or sets g.Config.Linux.Resources.BlockIO.ThrottleReadBpsDevice.
func (g *Generator) AddLinuxResourcesBlockIOThrottleReadBpsDevice(major int64, minor int64, rate uint64) {
	g.initConfigLinuxResourcesBlockIO()
	throttleDevices := addOrReplaceBlockIOThrottleDevice(g.Config.Linux.Resources.BlockIO.ThrottleReadBpsDevice, major, minor, rate)
	g.Config.Linux.Resources.BlockIO.ThrottleReadBpsDevice = throttleDevices
}

// DropLinuxResourcesBlockIOThrottleReadBpsDevice drops a item from g.Config.Linux.Resources.BlockIO.ThrottleReadBpsDevice.
func (g *Generator) DropLinuxResourcesBlockIOThrottleReadBpsDevice(major int64, minor int64) {
	if g.Config == nil || g.Config.Linux == nil || g.Config.Linux.Resources == nil || g.Config.Linux.Resources.BlockIO == nil {
		return
	}

	throttleDevices := dropBlockIOThrottleDevice(g.Config.Linux.Resources.BlockIO.ThrottleReadBpsDevice, major, minor)
	g.Config.Linux.Resources.BlockIO.ThrottleReadBpsDevice = throttleDevices
}

// AddLinuxResourcesBlockIOThrottleReadIOPSDevice adds or sets g.Config.Linux.Resources.BlockIO.ThrottleReadIOPSDevice.
func (g *Generator) AddLinuxResourcesBlockIOThrottleReadIOPSDevice(major int64, minor int64, rate uint64) {
	g.initConfigLinuxResourcesBlockIO()
	throttleDevices := addOrReplaceBlockIOThrottleDevice(g.Config.Linux.Resources.BlockIO.ThrottleReadIOPSDevice, major, minor, rate)
	g.Config.Linux.Resources.BlockIO.ThrottleReadIOPSDevice = throttleDevices
}

// DropLinuxResourcesBlockIOThrottleReadIOPSDevice drops a item from g.Config.Linux.Resources.BlockIO.ThrottleReadIOPSDevice.
func (g *Generator) DropLinuxResourcesBlockIOThrottleReadIOPSDevice(major int64, minor int64) {
	if g.Config == nil || g.Config.Linux == nil || g.Config.Linux.Resources == nil || g.Config.Linux.Resources.BlockIO == nil {
		return
	}

	throttleDevices := dropBlockIOThrottleDevice(g.Config.Linux.Resources.BlockIO.ThrottleReadIOPSDevice, major, minor)
	g.Config.Linux.Resources.BlockIO.ThrottleReadIOPSDevice = throttleDevices
}

// AddLinuxResourcesBlockIOThrottleWriteBpsDevice adds or sets g.Config.Linux.Resources.BlockIO.ThrottleWriteBpsDevice.
func (g *Generator) AddLinuxResourcesBlockIOThrottleWriteBpsDevice(major int64, minor int64, rate uint64) {
	g.initConfigLinuxResourcesBlockIO()
	throttleDevices := addOrReplaceBlockIOThrottleDevice(g.Config.Linux.Resources.BlockIO.ThrottleWriteBpsDevice, major, minor, rate)
	g.Config.Linux.Resources.BlockIO.ThrottleWriteBpsDevice = throttleDevices
}

// DropLinuxResourcesBlockIOThrottleWriteBpsDevice drops a item from g.Config.Linux.Resources.BlockIO.ThrottleWriteBpsDevice.
func (g *Generator) DropLinuxResourcesBlockIOThrottleWriteBpsDevice(major int64, minor int64) {
	if g.Config == nil || g.Config.Linux == nil || g.Config.Linux.Resources == nil || g.Config.Linux.Resources.BlockIO == nil {
		return
	}

	throttleDevices := dropBlockIOThrottleDevice(g.Config.Linux.Resources.BlockIO.ThrottleWriteBpsDevice, major, minor)
	g.Config.Linux.Resources.BlockIO.ThrottleWriteBpsDevice = throttleDevices
}

// AddLinuxResourcesBlockIOThrottleWriteIOPSDevice adds or sets g.Config.Linux.Resources.BlockIO.ThrottleWriteIOPSDevice.
func (g *Generator) AddLinuxResourcesBlockIOThrottleWriteIOPSDevice(major int64, minor int64, rate uint64) {
	g.initConfigLinuxResourcesBlockIO()
	throttleDevices := addOrReplaceBlockIOThrottleDevice(g.Config.Linux.Resources.BlockIO.ThrottleWriteIOPSDevice, major, minor, rate)
	g.Config.Linux.Resources.BlockIO.ThrottleWriteIOPSDevice = throttleDevices
}

// DropLinuxResourcesBlockIOThrottleWriteIOPSDevice drops a item from g.Config.Linux.Resources.BlockIO.ThrottleWriteIOPSDevice.
func (g *Generator) DropLinuxResourcesBlockIOThrottleWriteIOPSDevice(major int64, minor int64) {
	if g.Config == nil || g.Config.Linux == nil || g.Config.Linux.Resources == nil || g.Config.Linux.Resources.BlockIO == nil {
		return
	}

	throttleDevices := dropBlockIOThrottleDevice(g.Config.Linux.Resources.BlockIO.ThrottleWriteIOPSDevice, major, minor)
	g.Config.Linux.Resources.BlockIO.ThrottleWriteIOPSDevice = throttleDevices
}

// SetLinuxResourcesCPUShares sets g.Config.Linux.Resources.CPU.Shares.
func (g *Generator) SetLinuxResourcesCPUShares(shares uint64) {
	g.InitConfigLinuxResourcesCPU()
	g.Config.Linux.Resources.CPU.Shares = &shares
}

// SetLinuxResourcesCPUQuota sets g.Config.Linux.Resources.CPU.Quota.
func (g *Generator) SetLinuxResourcesCPUQuota(quota int64) {
	g.InitConfigLinuxResourcesCPU()
	g.Config.Linux.Resources.CPU.Quota = &quota
}

// SetLinuxResourcesCPUPeriod sets g.Config.Linux.Resources.CPU.Period.
func (g *Generator) SetLinuxResourcesCPUPeriod(period uint64) {
	g.InitConfigLinuxResourcesCPU()
	g.Config.Linux.Resources.CPU.Period = &period
}

// SetLinuxResourcesCPURealtimeRuntime sets g.Config.Linux.Resources.CPU.RealtimeRuntime.
func (g *Generator) SetLinuxResourcesCPURealtimeRuntime(time int64) {
	g.InitConfigLinuxResourcesCPU()
	g.Config.Linux.Resources.CPU.RealtimeRuntime = &time
}

// SetLinuxResourcesCPURealtimePeriod sets g.Config.Linux.Resources.CPU.RealtimePeriod.
func (g *Generator) SetLinuxResourcesCPURealtimePeriod(period uint64) {
	g.InitConfigLinuxResourcesCPU()
	g.Config.Linux.Resources.CPU.RealtimePeriod = &period
}

// SetLinuxResourcesCPUCpus sets g.Config.Linux.Resources.CPU.Cpus.
func (g *Generator) SetLinuxResourcesCPUCpus(cpus string) {
	g.InitConfigLinuxResourcesCPU()
	g.Config.Linux.Resources.CPU.Cpus = cpus
}

// SetLinuxResourcesCPUMems sets g.Config.Linux.Resources.CPU.Mems.
func (g *Generator) SetLinuxResourcesCPUMems(mems string) {
	g.InitConfigLinuxResourcesCPU()
	g.Config.Linux.Resources.CPU.Mems = mems
}

// AddLinuxResourcesHugepageLimit adds or sets g.Config.Linux.Resources.HugepageLimits.
func (g *Generator) AddLinuxResourcesHugepageLimit(pageSize string, limit uint64) {
	hugepageLimit := rspec.LinuxHugepageLimit{
		Pagesize: pageSize,
		Limit:    limit,
	}

	g.initConfigLinuxResources()
	for i, pageLimit := range g.Config.Linux.Resources.HugepageLimits {
		if pageLimit.Pagesize == pageSize {
			g.Config.Linux.Resources.HugepageLimits[i].Limit = limit
			return
		}
	}
	g.Config.Linux.Resources.HugepageLimits = append(g.Config.Linux.Resources.HugepageLimits, hugepageLimit)
}

// DropLinuxResourcesHugepageLimit drops a hugepage limit from g.Config.Linux.Resources.HugepageLimits.
func (g *Generator) DropLinuxResourcesHugepageLimit(pageSize string) {
	if g.Config == nil || g.Config.Linux == nil || g.Config.Linux.Resources == nil {
		return
	}

	for i, pageLimit := range g.Config.Linux.Resources.HugepageLimits {
		if pageLimit.Pagesize == pageSize {
			g.Config.Linux.Resources.HugepageLimits = append(g.Config.Linux.Resources.HugepageLimits[:i], g.Config.Linux.Resources.HugepageLimits[i+1:]...)
			return
		}
	}
}

// SetLinuxResourcesMemoryLimit sets g.Config.Linux.Resources.Memory.Limit.
func (g *Generator) SetLinuxResourcesMemoryLimit(limit int64) {
	g.initConfigLinuxResourcesMemory()
	g.Config.Linux.Resources.Memory.Limit = &limit
}

// SetLinuxResourcesMemoryReservation sets g.Config.Linux.Resources.Memory.Reservation.
func (g *Generator) SetLinuxResourcesMemoryReservation(reservation int64) {
	g.initConfigLinuxResourcesMemory()
	g.Config.Linux.Resources.Memory.Reservation = &reservation
}

// SetLinuxResourcesMemorySwap sets g.Config.Linux.Resources.Memory.Swap.
func (g *Generator) SetLinuxResourcesMemorySwap(swap int64) {
	g.initConfigLinuxResourcesMemory()
	g.Config.Linux.Resources.Memory.Swap = &swap
}

// SetLinuxResourcesMemoryKernel sets g.Config.Linux.Resources.Memory.Kernel.
func (g *Generator) SetLinuxResourcesMemoryKernel(kernel int64) {
	g.initConfigLinuxResourcesMemory()
	g.Config.Linux.Resources.Memory.Kernel = &kernel
}

// SetLinuxResourcesMemoryKernelTCP sets g.Config.Linux.Resources.Memory.KernelTCP.
func (g *Generator) SetLinuxResourcesMemoryKernelTCP(kernelTCP int64) {
	g.initConfigLinuxResourcesMemory()
	g.Config.Linux.Resources.Memory.KernelTCP = &kernelTCP
}

// SetLinuxResourcesMemorySwappiness sets g.Config.Linux.Resources.Memory.Swappiness.
func (g *Generator) SetLinuxResourcesMemorySwappiness(swappiness uint64) {
	g.initConfigLinuxResourcesMemory()
	g.Config.Linux.Resources.Memory.Swappiness = &swappiness
}

// SetLinuxResourcesMemoryDisableOOMKiller sets g.Config.Linux.Resources.Memory.DisableOOMKiller.
func (g *Generator) SetLinuxResourcesMemoryDisableOOMKiller(disable bool) {
	g.initConfigLinuxResourcesMemory()
	g.Config.Linux.Resources.Memory.DisableOOMKiller = &disable
}

// SetLinuxResourcesNetworkClassID sets g.Config.Linux.Resources.Network.ClassID.
func (g *Generator) SetLinuxResourcesNetworkClassID(classid uint32) {
	g.initConfigLinuxResourcesNetwork()
	g.Config.Linux.Resources.Network.ClassID = &classid
}

// AddLinuxResourcesNetworkPriorities adds or sets g.Config.Linux.Resources.Network.Priorities.
func (g *Generator) AddLinuxResourcesNetworkPriorities(name string, prio uint32) {
	g.initConfigLinuxResourcesNetwork()
	for i, netPriority := range g.Config.Linux.Resources.Network.Priorities {
		if netPriority.Name == name {
			g.Config.Linux.Resources.Network.Priorities[i].Priority = prio
			return
		}
	}
	interfacePrio := new(rspec.LinuxInterfacePriority)
	interfacePrio.Name = name
	interfacePrio.Priority = prio
	g.Config.Linux.Resources.Network.Priorities = append(g.Config.Linux.Resources.Network.Priorities, *interfacePrio)
}

// DropLinuxResourcesNetworkPriorities drops one item from g.Config.Linux.Resources.Network.Priorities.
func (g *Generator) DropLinuxResourcesNetworkPriorities(name string) {
	if g.Config == nil || g.Config.Linux == nil || g.Config.Linux.Resources == nil || g.Config.Linux.Resources.Network == nil {
		return
	}

	for i, netPriority := range g.Config.Linux.Resources.Network.Priorities {
		if netPriority.Name == name {
			g.Config.Linux.Resources.Network.Priorities = append(g.Config.Linux.Resources.Network.Priorities[:i], g.Config.Linux.Resources.Network.Priorities[i+1:]...)
			return
		}
	}
}

// SetLinuxResourcesPidsLimit sets g.Config.Linux.Resources.Pids.Limit.
func (g *Generator) SetLinuxResourcesPidsLimit(limit int64) {
	g.initConfigLinuxResourcesPids()
	g.Config.Linux.Resources.Pids.Limit = limit
}

// ClearLinuxSysctl clears g.Config.Linux.Sysctl.
func (g *Generator) ClearLinuxSysctl() {
	if g.Config == nil || g.Config.Linux == nil {
		return
	}
	g.Config.Linux.Sysctl = make(map[string]string)
}

// AddLinuxSysctl adds a new sysctl config into g.Config.Linux.Sysctl.
func (g *Generator) AddLinuxSysctl(key, value string) {
	g.initConfigLinuxSysctl()
	g.Config.Linux.Sysctl[key] = value
}

// RemoveLinuxSysctl removes a sysctl config from g.Config.Linux.Sysctl.
func (g *Generator) RemoveLinuxSysctl(key string) {
	if g.Config == nil || g.Config.Linux == nil || g.Config.Linux.Sysctl == nil {
		return
	}
	delete(g.Config.Linux.Sysctl, key)
}

// ClearLinuxUIDMappings clear g.Config.Linux.UIDMappings.
func (g *Generator) ClearLinuxUIDMappings() {
	if g.Config == nil || g.Config.Linux == nil {
		return
	}
	g.Config.Linux.UIDMappings = []rspec.LinuxIDMapping{}
}

// AddLinuxUIDMapping adds uidMap into g.Config.Linux.UIDMappings.
func (g *Generator) AddLinuxUIDMapping(hid, cid, size uint32) {
	idMapping := rspec.LinuxIDMapping{
		HostID:      hid,
		ContainerID: cid,
		Size:        size,
	}

	g.initConfigLinux()
	g.Config.Linux.UIDMappings = append(g.Config.Linux.UIDMappings, idMapping)
}

// ClearLinuxGIDMappings clear g.Config.Linux.GIDMappings.
func (g *Generator) ClearLinuxGIDMappings() {
	if g.Config == nil || g.Config.Linux == nil {
		return
	}
	g.Config.Linux.GIDMappings = []rspec.LinuxIDMapping{}
}

// AddLinuxGIDMapping adds gidMap into g.Config.Linux.GIDMappings.
func (g *Generator) AddLinuxGIDMapping(hid, cid, size uint32) {
	idMapping := rspec.LinuxIDMapping{
		HostID:      hid,
		ContainerID: cid,
		Size:        size,
	}

	g.initConfigLinux()
	g.Config.Linux.GIDMappings = append(g.Config.Linux.GIDMappings, idMapping)
}

// SetLinuxRootPropagation sets g.Config.Linux.RootfsPropagation.
func (g *Generator) SetLinuxRootPropagation(rp string) error {
	switch rp {
	case "":
	case "private":
	case "rprivate":
	case "slave":
	case "rslave":
	case "shared":
	case "rshared":
	case "unbindable":
	case "runbindable":
	default:
		return fmt.Errorf("rootfs-propagation %q must be empty or one of (r)private|(r)slave|(r)shared|(r)unbindable", rp)
	}
	g.initConfigLinux()
	g.Config.Linux.RootfsPropagation = rp
	return nil
}

// ClearPreStartHooks clear g.Config.Hooks.Prestart.
func (g *Generator) ClearPreStartHooks() {
	if g.Config == nil || g.Config.Hooks == nil {
		return
	}
	g.Config.Hooks.Prestart = []rspec.Hook{}
}

// AddPreStartHook add a prestart hook into g.Config.Hooks.Prestart.
func (g *Generator) AddPreStartHook(preStartHook rspec.Hook) error {
	g.initConfigHooks()
	g.Config.Hooks.Prestart = append(g.Config.Hooks.Prestart, preStartHook)
	return nil
}

// ClearPostStopHooks clear g.Config.Hooks.Poststop.
func (g *Generator) ClearPostStopHooks() {
	if g.Config == nil || g.Config.Hooks == nil {
		return
	}
	g.Config.Hooks.Poststop = []rspec.Hook{}
}

// AddPostStopHook adds a poststop hook into g.Config.Hooks.Poststop.
func (g *Generator) AddPostStopHook(postStopHook rspec.Hook) error {
	g.initConfigHooks()
	g.Config.Hooks.Poststop = append(g.Config.Hooks.Poststop, postStopHook)
	return nil
}

// ClearPostStartHooks clear g.Config.Hooks.Poststart.
func (g *Generator) ClearPostStartHooks() {
	if g.Config == nil || g.Config.Hooks == nil {
		return
	}
	g.Config.Hooks.Poststart = []rspec.Hook{}
}

// AddPostStartHook adds a poststart hook into g.Config.Hooks.Poststart.
func (g *Generator) AddPostStartHook(postStartHook rspec.Hook) error {
	g.initConfigHooks()
	g.Config.Hooks.Poststart = append(g.Config.Hooks.Poststart, postStartHook)
	return nil
}

// AddMount adds a mount into g.Config.Mounts.
func (g *Generator) AddMount(mnt rspec.Mount) {
	g.initConfig()

	g.Config.Mounts = append(g.Config.Mounts, mnt)
}

// RemoveMount removes a mount point on the dest directory
func (g *Generator) RemoveMount(dest string) {
	g.initConfig()

	for index, mount := range g.Config.Mounts {
		if mount.Destination == dest {
			g.Config.Mounts = append(g.Config.Mounts[:index], g.Config.Mounts[index+1:]...)
			return
		}
	}
}

// Mounts returns the list of mounts
func (g *Generator) Mounts() []rspec.Mount {
	g.initConfig()

	return g.Config.Mounts
}

// ClearMounts clear g.Config.Mounts
func (g *Generator) ClearMounts() {
	if g.Config == nil {
		return
	}
	g.Config.Mounts = []rspec.Mount{}
}

// SetupPrivileged sets up the privilege-related fields inside g.Config.
func (g *Generator) SetupPrivileged(privileged bool) {
	if privileged { // Add all capabilities in privileged mode.
		var finalCapList []string
		for _, cap := range capability.List() {
			if g.HostSpecific && cap > validate.LastCap() {
				continue
			}
			finalCapList = append(finalCapList, fmt.Sprintf("CAP_%s", strings.ToUpper(cap.String())))
		}
		g.initConfigLinux()
		g.initConfigProcessCapabilities()
		g.ClearProcessCapabilities()
		g.Config.Process.Capabilities.Bounding = append(g.Config.Process.Capabilities.Bounding, finalCapList...)
		g.Config.Process.Capabilities.Effective = append(g.Config.Process.Capabilities.Effective, finalCapList...)
		g.Config.Process.Capabilities.Inheritable = append(g.Config.Process.Capabilities.Inheritable, finalCapList...)
		g.Config.Process.Capabilities.Permitted = append(g.Config.Process.Capabilities.Permitted, finalCapList...)
		g.Config.Process.Capabilities.Ambient = append(g.Config.Process.Capabilities.Ambient, finalCapList...)
		g.Config.Process.SelinuxLabel = ""
		g.Config.Process.ApparmorProfile = ""
		g.Config.Linux.Seccomp = nil
	}
}

// ClearProcessCapabilities clear g.Config.Process.Capabilities.
func (g *Generator) ClearProcessCapabilities() {
	if g.Config == nil || g.Config.Process == nil || g.Config.Process.Capabilities == nil {
		return
	}
	g.Config.Process.Capabilities.Bounding = []string{}
	g.Config.Process.Capabilities.Effective = []string{}
	g.Config.Process.Capabilities.Inheritable = []string{}
	g.Config.Process.Capabilities.Permitted = []string{}
	g.Config.Process.Capabilities.Ambient = []string{}
}

// AddProcessCapability adds a process capability into all 5 capability sets.
func (g *Generator) AddProcessCapability(c string) error {
	cp := strings.ToUpper(c)
	if err := validate.CapValid(cp, g.HostSpecific); err != nil {
		return err
	}

	g.initConfigProcessCapabilities()

	var foundAmbient, foundBounding, foundEffective, foundInheritable, foundPermitted bool
	for _, cap := range g.Config.Process.Capabilities.Ambient {
		if strings.ToUpper(cap) == cp {
			foundAmbient = true
			break
		}
	}
	if !foundAmbient {
		g.Config.Process.Capabilities.Ambient = append(g.Config.Process.Capabilities.Ambient, cp)
	}

	for _, cap := range g.Config.Process.Capabilities.Bounding {
		if strings.ToUpper(cap) == cp {
			foundBounding = true
			break
		}
	}
	if !foundBounding {
		g.Config.Process.Capabilities.Bounding = append(g.Config.Process.Capabilities.Bounding, cp)
	}

	for _, cap := range g.Config.Process.Capabilities.Effective {
		if strings.ToUpper(cap) == cp {
			foundEffective = true
			break
		}
	}
	if !foundEffective {
		g.Config.Process.Capabilities.Effective = append(g.Config.Process.Capabilities.Effective, cp)
	}

	for _, cap := range g.Config.Process.Capabilities.Inheritable {
		if strings.ToUpper(cap) == cp {
			foundInheritable = true
			break
		}
	}
	if !foundInheritable {
		g.Config.Process.Capabilities.Inheritable = append(g.Config.Process.Capabilities.Inheritable, cp)
	}

	for _, cap := range g.Config.Process.Capabilities.Permitted {
		if strings.ToUpper(cap) == cp {
			foundPermitted = true
			break
		}
	}
	if !foundPermitted {
		g.Config.Process.Capabilities.Permitted = append(g.Config.Process.Capabilities.Permitted, cp)
	}

	return nil
}

// AddProcessCapabilityAmbient adds a process capability into g.Config.Process.Capabilities.Ambient.
func (g *Generator) AddProcessCapabilityAmbient(c string) error {
	cp := strings.ToUpper(c)
	if err := validate.CapValid(cp, g.HostSpecific); err != nil {
		return err
	}

	g.initConfigProcessCapabilities()

	var foundAmbient bool
	for _, cap := range g.Config.Process.Capabilities.Ambient {
		if strings.ToUpper(cap) == cp {
			foundAmbient = true
			break
		}
	}

	if !foundAmbient {
		g.Config.Process.Capabilities.Ambient = append(g.Config.Process.Capabilities.Ambient, cp)
	}

	return nil
}

// AddProcessCapabilityBounding adds a process capability into g.Config.Process.Capabilities.Bounding.
func (g *Generator) AddProcessCapabilityBounding(c string) error {
	cp := strings.ToUpper(c)
	if err := validate.CapValid(cp, g.HostSpecific); err != nil {
		return err
	}

	g.initConfigProcessCapabilities()

	var foundBounding bool
	for _, cap := range g.Config.Process.Capabilities.Bounding {
		if strings.ToUpper(cap) == cp {
			foundBounding = true
			break
		}
	}
	if !foundBounding {
		g.Config.Process.Capabilities.Bounding = append(g.Config.Process.Capabilities.Bounding, cp)
	}

	return nil
}

// AddProcessCapabilityEffective adds a process capability into g.Config.Process.Capabilities.Effective.
func (g *Generator) AddProcessCapabilityEffective(c string) error {
	cp := strings.ToUpper(c)
	if err := validate.CapValid(cp, g.HostSpecific); err != nil {
		return err
	}

	g.initConfigProcessCapabilities()

	var foundEffective bool
	for _, cap := range g.Config.Process.Capabilities.Effective {
		if strings.ToUpper(cap) == cp {
			foundEffective = true
			break
		}
	}
	if !foundEffective {
		g.Config.Process.Capabilities.Effective = append(g.Config.Process.Capabilities.Effective, cp)
	}

	return nil
}

// AddProcessCapabilityInheritable adds a process capability into g.Config.Process.Capabilities.Inheritable.
func (g *Generator) AddProcessCapabilityInheritable(c string) error {
	cp := strings.ToUpper(c)
	if err := validate.CapValid(cp, g.HostSpecific); err != nil {
		return err
	}

	g.initConfigProcessCapabilities()

	var foundInheritable bool
	for _, cap := range g.Config.Process.Capabilities.Inheritable {
		if strings.ToUpper(cap) == cp {
			foundInheritable = true
			break
		}
	}
	if !foundInheritable {
		g.Config.Process.Capabilities.Inheritable = append(g.Config.Process.Capabilities.Inheritable, cp)
	}

	return nil
}

// AddProcessCapabilityPermitted adds a process capability into g.Config.Process.Capabilities.Permitted.
func (g *Generator) AddProcessCapabilityPermitted(c string) error {
	cp := strings.ToUpper(c)
	if err := validate.CapValid(cp, g.HostSpecific); err != nil {
		return err
	}

	g.initConfigProcessCapabilities()

	var foundPermitted bool
	for _, cap := range g.Config.Process.Capabilities.Permitted {
		if strings.ToUpper(cap) == cp {
			foundPermitted = true
			break
		}
	}
	if !foundPermitted {
		g.Config.Process.Capabilities.Permitted = append(g.Config.Process.Capabilities.Permitted, cp)
	}

	return nil
}

// DropProcessCapability drops a process capability from all 5 capability sets.
func (g *Generator) DropProcessCapability(c string) error {
	if g.Config == nil || g.Config.Process == nil || g.Config.Process.Capabilities == nil {
		return nil
	}

	cp := strings.ToUpper(c)
	for i, cap := range g.Config.Process.Capabilities.Ambient {
		if strings.ToUpper(cap) == cp {
			g.Config.Process.Capabilities.Ambient = removeFunc(g.Config.Process.Capabilities.Ambient, i)
		}
	}
	for i, cap := range g.Config.Process.Capabilities.Bounding {
		if strings.ToUpper(cap) == cp {
			g.Config.Process.Capabilities.Bounding = removeFunc(g.Config.Process.Capabilities.Bounding, i)
		}
	}
	for i, cap := range g.Config.Process.Capabilities.Effective {
		if strings.ToUpper(cap) == cp {
			g.Config.Process.Capabilities.Effective = removeFunc(g.Config.Process.Capabilities.Effective, i)
		}
	}
	for i, cap := range g.Config.Process.Capabilities.Inheritable {
		if strings.ToUpper(cap) == cp {
			g.Config.Process.Capabilities.Inheritable = removeFunc(g.Config.Process.Capabilities.Inheritable, i)
		}
	}
	for i, cap := range g.Config.Process.Capabilities.Permitted {
		if strings.ToUpper(cap) == cp {
			g.Config.Process.Capabilities.Permitted = removeFunc(g.Config.Process.Capabilities.Permitted, i)
		}
	}

	return validate.CapValid(cp, false)
}

// DropProcessCapabilityAmbient drops a process capability from g.Config.Process.Capabilities.Ambient.
func (g *Generator) DropProcessCapabilityAmbient(c string) error {
	if g.Config == nil || g.Config.Process == nil || g.Config.Process.Capabilities == nil {
		return nil
	}

	cp := strings.ToUpper(c)
	for i, cap := range g.Config.Process.Capabilities.Ambient {
		if strings.ToUpper(cap) == cp {
			g.Config.Process.Capabilities.Ambient = removeFunc(g.Config.Process.Capabilities.Ambient, i)
		}
	}

	return validate.CapValid(cp, false)
}

// DropProcessCapabilityBounding drops a process capability from g.Config.Process.Capabilities.Bounding.
func (g *Generator) DropProcessCapabilityBounding(c string) error {
	if g.Config == nil || g.Config.Process == nil || g.Config.Process.Capabilities == nil {
		return nil
	}

	cp := strings.ToUpper(c)
	for i, cap := range g.Config.Process.Capabilities.Bounding {
		if strings.ToUpper(cap) == cp {
			g.Config.Process.Capabilities.Bounding = removeFunc(g.Config.Process.Capabilities.Bounding, i)
		}
	}

	return validate.CapValid(cp, false)
}

// DropProcessCapabilityEffective drops a process capability from g.Config.Process.Capabilities.Effective.
func (g *Generator) DropProcessCapabilityEffective(c string) error {
	if g.Config == nil || g.Config.Process == nil || g.Config.Process.Capabilities == nil {
		return nil
	}

	cp := strings.ToUpper(c)
	for i, cap := range g.Config.Process.Capabilities.Effective {
		if strings.ToUpper(cap) == cp {
			g.Config.Process.Capabilities.Effective = removeFunc(g.Config.Process.Capabilities.Effective, i)
		}
	}

	return validate.CapValid(cp, false)
}

// DropProcessCapabilityInheritable drops a process capability from g.Config.Process.Capabilities.Inheritable.
func (g *Generator) DropProcessCapabilityInheritable(c string) error {
	if g.Config == nil || g.Config.Process == nil || g.Config.Process.Capabilities == nil {
		return nil
	}

	cp := strings.ToUpper(c)
	for i, cap := range g.Config.Process.Capabilities.Inheritable {
		if strings.ToUpper(cap) == cp {
			g.Config.Process.Capabilities.Inheritable = removeFunc(g.Config.Process.Capabilities.Inheritable, i)
		}
	}

	return validate.CapValid(cp, false)
}

// DropProcessCapabilityPermitted drops a process capability from g.Config.Process.Capabilities.Permitted.
func (g *Generator) DropProcessCapabilityPermitted(c string) error {
	if g.Config == nil || g.Config.Process == nil || g.Config.Process.Capabilities == nil {
		return nil
	}

	cp := strings.ToUpper(c)
	for i, cap := range g.Config.Process.Capabilities.Permitted {
		if strings.ToUpper(cap) == cp {
			g.Config.Process.Capabilities.Permitted = removeFunc(g.Config.Process.Capabilities.Permitted, i)
		}
	}

	return validate.CapValid(cp, false)
}

func mapStrToNamespace(ns string, path string) (rspec.LinuxNamespace, error) {
	switch ns {
	case "network":
		return rspec.LinuxNamespace{Type: rspec.NetworkNamespace, Path: path}, nil
	case "pid":
		return rspec.LinuxNamespace{Type: rspec.PIDNamespace, Path: path}, nil
	case "mount":
		return rspec.LinuxNamespace{Type: rspec.MountNamespace, Path: path}, nil
	case "ipc":
		return rspec.LinuxNamespace{Type: rspec.IPCNamespace, Path: path}, nil
	case "uts":
		return rspec.LinuxNamespace{Type: rspec.UTSNamespace, Path: path}, nil
	case "user":
		return rspec.LinuxNamespace{Type: rspec.UserNamespace, Path: path}, nil
	case "cgroup":
		return rspec.LinuxNamespace{Type: rspec.CgroupNamespace, Path: path}, nil
	default:
		return rspec.LinuxNamespace{}, fmt.Errorf("unrecognized namespace %q", ns)
	}
}

// ClearLinuxNamespaces clear g.Config.Linux.Namespaces.
func (g *Generator) ClearLinuxNamespaces() {
	if g.Config == nil || g.Config.Linux == nil {
		return
	}
	g.Config.Linux.Namespaces = []rspec.LinuxNamespace{}
}

// AddOrReplaceLinuxNamespace adds or replaces a namespace inside
// g.Config.Linux.Namespaces.
func (g *Generator) AddOrReplaceLinuxNamespace(ns string, path string) error {
	namespace, err := mapStrToNamespace(ns, path)
	if err != nil {
		return err
	}

	g.initConfigLinux()
	for i, ns := range g.Config.Linux.Namespaces {
		if ns.Type == namespace.Type {
			g.Config.Linux.Namespaces[i] = namespace
			return nil
		}
	}
	g.Config.Linux.Namespaces = append(g.Config.Linux.Namespaces, namespace)
	return nil
}

// RemoveLinuxNamespace removes a namespace from g.Config.Linux.Namespaces.
func (g *Generator) RemoveLinuxNamespace(ns string) error {
	namespace, err := mapStrToNamespace(ns, "")
	if err != nil {
		return err
	}

	if g.Config == nil || g.Config.Linux == nil {
		return nil
	}
	for i, ns := range g.Config.Linux.Namespaces {
		if ns.Type == namespace.Type {
			g.Config.Linux.Namespaces = append(g.Config.Linux.Namespaces[:i], g.Config.Linux.Namespaces[i+1:]...)
			return nil
		}
	}
	return nil
}

// AddDevice - add a device into g.Config.Linux.Devices
func (g *Generator) AddDevice(device rspec.LinuxDevice) {
	g.initConfigLinux()

	for i, dev := range g.Config.Linux.Devices {
		if dev.Path == device.Path {
			g.Config.Linux.Devices[i] = device
			return
		}
		if dev.Type == device.Type && dev.Major == device.Major && dev.Minor == device.Minor {
			fmt.Fprintln(os.Stderr, "WARNING: The same type, major and minor should not be used for multiple devices.")
		}
	}

	g.Config.Linux.Devices = append(g.Config.Linux.Devices, device)
}

// RemoveDevice remove a device from g.Config.Linux.Devices
func (g *Generator) RemoveDevice(path string) {
	if g.Config == nil || g.Config.Linux == nil || g.Config.Linux.Devices == nil {
		return
	}

	for i, device := range g.Config.Linux.Devices {
		if device.Path == path {
			g.Config.Linux.Devices = append(g.Config.Linux.Devices[:i], g.Config.Linux.Devices[i+1:]...)
			return
		}
	}
}

// ClearLinuxDevices clears g.Config.Linux.Devices
func (g *Generator) ClearLinuxDevices() {
	if g.Config == nil || g.Config.Linux == nil || g.Config.Linux.Devices == nil {
		return
	}

	g.Config.Linux.Devices = []rspec.LinuxDevice{}
}

// AddLinuxResourcesDevice - add a device into g.Config.Linux.Resources.Devices
func (g *Generator) AddLinuxResourcesDevice(allow bool, devType string, major, minor *int64, access string) {
	g.initConfigLinuxResources()

	device := rspec.LinuxDeviceCgroup{
		Allow:  allow,
		Type:   devType,
		Access: access,
		Major:  major,
		Minor:  minor,
	}
	g.Config.Linux.Resources.Devices = append(g.Config.Linux.Resources.Devices, device)
}

// RemoveLinuxResourcesDevice - remove a device from g.Config.Linux.Resources.Devices
func (g *Generator) RemoveLinuxResourcesDevice(allow bool, devType string, major, minor *int64, access string) {
	if g.Config == nil || g.Config.Linux == nil || g.Config.Linux.Resources == nil {
		return
	}
	for i, device := range g.Config.Linux.Resources.Devices {
		if device.Allow == allow &&
			(devType == device.Type || (devType != "" && device.Type != "" && devType == device.Type)) &&
			(access == device.Access || (access != "" && device.Access != "" && access == device.Access)) &&
			(major == device.Major || (major != nil && device.Major != nil && *major == *device.Major)) &&
			(minor == device.Minor || (minor != nil && device.Minor != nil && *minor == *device.Minor)) {

			g.Config.Linux.Resources.Devices = append(g.Config.Linux.Resources.Devices[:i], g.Config.Linux.Resources.Devices[i+1:]...)
			return
		}
	}
	return
}

// strPtr returns the pointer pointing to the string s.
func strPtr(s string) *string { return &s }

// SetSyscallAction adds rules for syscalls with the specified action
func (g *Generator) SetSyscallAction(arguments seccomp.SyscallOpts) error {
	g.initConfigLinuxSeccomp()
	return seccomp.ParseSyscallFlag(arguments, g.Config.Linux.Seccomp)
}

// SetDefaultSeccompAction sets the default action for all syscalls not defined
// and then removes any syscall rules with this action already specified.
func (g *Generator) SetDefaultSeccompAction(action string) error {
	g.initConfigLinuxSeccomp()
	return seccomp.ParseDefaultAction(action, g.Config.Linux.Seccomp)
}

// SetDefaultSeccompActionForce only sets the default action for all syscalls not defined
func (g *Generator) SetDefaultSeccompActionForce(action string) error {
	g.initConfigLinuxSeccomp()
	return seccomp.ParseDefaultActionForce(action, g.Config.Linux.Seccomp)
}

// SetSeccompArchitecture sets the supported seccomp architectures
func (g *Generator) SetSeccompArchitecture(architecture string) error {
	g.initConfigLinuxSeccomp()
	return seccomp.ParseArchitectureFlag(architecture, g.Config.Linux.Seccomp)
}

// RemoveSeccompRule removes rules for any specified syscalls
func (g *Generator) RemoveSeccompRule(arguments string) error {
	g.initConfigLinuxSeccomp()
	return seccomp.RemoveAction(arguments, g.Config.Linux.Seccomp)
}

// RemoveAllSeccompRules removes all syscall rules
func (g *Generator) RemoveAllSeccompRules() error {
	g.initConfigLinuxSeccomp()
	return seccomp.RemoveAllSeccompRules(g.Config.Linux.Seccomp)
}

// AddLinuxMaskedPaths adds masked paths into g.Config.Linux.MaskedPaths.
func (g *Generator) AddLinuxMaskedPaths(path string) {
	g.initConfigLinux()
	g.Config.Linux.MaskedPaths = append(g.Config.Linux.MaskedPaths, path)
}

// AddLinuxReadonlyPaths adds readonly paths into g.Config.Linux.MaskedPaths.
func (g *Generator) AddLinuxReadonlyPaths(path string) {
	g.initConfigLinux()
	g.Config.Linux.ReadonlyPaths = append(g.Config.Linux.ReadonlyPaths, path)
}

func addOrReplaceBlockIOThrottleDevice(tmpList []rspec.LinuxThrottleDevice, major int64, minor int64, rate uint64) []rspec.LinuxThrottleDevice {
	throttleDevices := tmpList
	for i, throttleDevice := range throttleDevices {
		if throttleDevice.Major == major && throttleDevice.Minor == minor {
			throttleDevices[i].Rate = rate
			return throttleDevices
		}
	}
	throttleDevice := new(rspec.LinuxThrottleDevice)
	throttleDevice.Major = major
	throttleDevice.Minor = minor
	throttleDevice.Rate = rate
	throttleDevices = append(throttleDevices, *throttleDevice)

	return throttleDevices
}

func dropBlockIOThrottleDevice(tmpList []rspec.LinuxThrottleDevice, major int64, minor int64) []rspec.LinuxThrottleDevice {
	throttleDevices := tmpList
	for i, throttleDevice := range throttleDevices {
		if throttleDevice.Major == major && throttleDevice.Minor == minor {
			throttleDevices = append(throttleDevices[:i], throttleDevices[i+1:]...)
			return throttleDevices
		}
	}

	return throttleDevices
}

// AddSolarisAnet adds network into g.Config.Solaris.Anet
func (g *Generator) AddSolarisAnet(anet rspec.SolarisAnet) {
	g.initConfigSolaris()
	g.Config.Solaris.Anet = append(g.Config.Solaris.Anet, anet)
}

// SetSolarisCappedCPUNcpus sets g.Config.Solaris.CappedCPU.Ncpus
func (g *Generator) SetSolarisCappedCPUNcpus(ncpus string) {
	g.initConfigSolarisCappedCPU()
	g.Config.Solaris.CappedCPU.Ncpus = ncpus
}

// SetSolarisCappedMemoryPhysical sets g.Config.Solaris.CappedMemory.Physical
func (g *Generator) SetSolarisCappedMemoryPhysical(physical string) {
	g.initConfigSolarisCappedMemory()
	g.Config.Solaris.CappedMemory.Physical = physical
}

// SetSolarisCappedMemorySwap sets g.Config.Solaris.CappedMemory.Swap
func (g *Generator) SetSolarisCappedMemorySwap(swap string) {
	g.initConfigSolarisCappedMemory()
	g.Config.Solaris.CappedMemory.Swap = swap
}

// SetSolarisLimitPriv sets g.Config.Solaris.LimitPriv
func (g *Generator) SetSolarisLimitPriv(limitPriv string) {
	g.initConfigSolaris()
	g.Config.Solaris.LimitPriv = limitPriv
}

// SetSolarisMaxShmMemory sets g.Config.Solaris.MaxShmMemory
func (g *Generator) SetSolarisMaxShmMemory(memory string) {
	g.initConfigSolaris()
	g.Config.Solaris.MaxShmMemory = memory
}

// SetSolarisMilestone sets g.Config.Solaris.Milestone
func (g *Generator) SetSolarisMilestone(milestone string) {
	g.initConfigSolaris()
	g.Config.Solaris.Milestone = milestone
}

// SetVMHypervisorPath sets g.Config.VM.Hypervisor.Path
func (g *Generator) SetVMHypervisorPath(path string) error {
	if !strings.HasPrefix(path, "/") {
		return fmt.Errorf("hypervisorPath %v is not an absolute path", path)
	}
	g.initConfigVMHypervisor()
	g.Config.VM.Hypervisor.Path = path
	return nil
}

// SetVMHypervisorParameters sets g.Config.VM.Hypervisor.Parameters
func (g *Generator) SetVMHypervisorParameters(parameters []string) {
	g.initConfigVMHypervisor()
	g.Config.VM.Hypervisor.Parameters = parameters
}

// SetVMKernelPath sets g.Config.VM.Kernel.Path
func (g *Generator) SetVMKernelPath(path string) error {
	if !strings.HasPrefix(path, "/") {
		return fmt.Errorf("kernelPath %v is not an absolute path", path)
	}
	g.initConfigVMKernel()
	g.Config.VM.Kernel.Path = path
	return nil
}

// SetVMKernelParameters sets g.Config.VM.Kernel.Parameters
func (g *Generator) SetVMKernelParameters(parameters []string) {
	g.initConfigVMKernel()
	g.Config.VM.Kernel.Parameters = parameters
}

// SetVMKernelInitRD sets g.Config.VM.Kernel.InitRD
func (g *Generator) SetVMKernelInitRD(initrd string) error {
	if !strings.HasPrefix(initrd, "/") {
		return fmt.Errorf("kernelInitrd %v is not an absolute path", initrd)
	}
	g.initConfigVMKernel()
	g.Config.VM.Kernel.InitRD = initrd
	return nil
}

// SetVMImagePath sets g.Config.VM.Image.Path
func (g *Generator) SetVMImagePath(path string) error {
	if !strings.HasPrefix(path, "/") {
		return fmt.Errorf("imagePath %v is not an absolute path", path)
	}
	g.initConfigVMImage()
	g.Config.VM.Image.Path = path
	return nil
}

// SetVMImageFormat sets g.Config.VM.Image.Format
func (g *Generator) SetVMImageFormat(format string) error {
	switch format {
	case "raw":
	case "qcow2":
	case "vdi":
	case "vmdk":
	case "vhd":
	default:
		return fmt.Errorf("Commonly supported formats are: raw, qcow2, vdi, vmdk, vhd")
	}
	g.initConfigVMImage()
	g.Config.VM.Image.Format = format
	return nil
}

// SetWindowsHypervUntilityVMPath sets g.Config.Windows.HyperV.UtilityVMPath.
func (g *Generator) SetWindowsHypervUntilityVMPath(path string) {
	g.initConfigWindowsHyperV()
	g.Config.Windows.HyperV.UtilityVMPath = path
}

// SetWindowsIgnoreFlushesDuringBoot sets g.Config.Windows.IgnoreFlushesDuringBoot.
func (g *Generator) SetWindowsIgnoreFlushesDuringBoot(ignore bool) {
	g.initConfigWindows()
	g.Config.Windows.IgnoreFlushesDuringBoot = ignore
}

// AddWindowsLayerFolders adds layer folders into  g.Config.Windows.LayerFolders.
func (g *Generator) AddWindowsLayerFolders(folder string) {
	g.initConfigWindows()
	g.Config.Windows.LayerFolders = append(g.Config.Windows.LayerFolders, folder)
}

// AddWindowsDevices adds or sets g.Config.Windwos.Devices
func (g *Generator) AddWindowsDevices(id, idType string) error {
	if idType != "class" {
		return fmt.Errorf("Invalid idType value: %s. Windows only supports a value of class", idType)
	}
	device := rspec.WindowsDevice{
		ID:     id,
		IDType: idType,
	}

	g.initConfigWindows()
	for i, device := range g.Config.Windows.Devices {
		if device.ID == id {
			g.Config.Windows.Devices[i].IDType = idType
			return nil
		}
	}
	g.Config.Windows.Devices = append(g.Config.Windows.Devices, device)
	return nil
}

// SetWindowsNetwork sets g.Config.Windows.Network.
func (g *Generator) SetWindowsNetwork(network rspec.WindowsNetwork) {
	g.initConfigWindows()
	g.Config.Windows.Network = &network
}

// SetWindowsNetworkAllowUnqualifiedDNSQuery sets g.Config.Windows.Network.AllowUnqualifiedDNSQuery
func (g *Generator) SetWindowsNetworkAllowUnqualifiedDNSQuery(setting bool) {
	g.initConfigWindowsNetwork()
	g.Config.Windows.Network.AllowUnqualifiedDNSQuery = setting
}

// SetWindowsNetworkNamespace sets g.Config.Windows.Network.NetworkNamespace
func (g *Generator) SetWindowsNetworkNamespace(path string) {
	g.initConfigWindowsNetwork()
	g.Config.Windows.Network.NetworkNamespace = path
}

// SetWindowsResourcesCPU sets g.Config.Windows.Resources.CPU.
func (g *Generator) SetWindowsResourcesCPU(cpu rspec.WindowsCPUResources) {
	g.initConfigWindowsResources()
	g.Config.Windows.Resources.CPU = &cpu
}

// SetWindowsResourcesMemoryLimit sets g.Config.Windows.Resources.Memory.Limit.
func (g *Generator) SetWindowsResourcesMemoryLimit(limit uint64) {
	g.initConfigWindowsResourcesMemory()
	g.Config.Windows.Resources.Memory.Limit = &limit
}

// SetWindowsResourcesStorage sets g.Config.Windows.Resources.Storage.
func (g *Generator) SetWindowsResourcesStorage(storage rspec.WindowsStorageResources) {
	g.initConfigWindowsResources()
	g.Config.Windows.Resources.Storage = &storage
}

// SetWindowsServicing sets g.Config.Windows.Servicing.
func (g *Generator) SetWindowsServicing(servicing bool) {
	g.initConfigWindows()
	g.Config.Windows.Servicing = servicing
}
