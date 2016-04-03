// +build linux

package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"

	"github.com/Sirupsen/logrus"
	"github.com/codegangsta/cli"
	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runc/libcontainer/seccomp"
	"github.com/opencontainers/specs"
)

var specCommand = cli.Command{
	Name:  "spec",
	Usage: "create a new specification file",
	Flags: []cli.Flag{
		cli.StringFlag{
			Name:  "bundle, b",
			Value: "",
			Usage: "path to the root of the bundle directory",
		},
	},
	Action: func(context *cli.Context) {
		spec := specs.LinuxSpec{
			Spec: specs.Spec{
				Version: specs.Version,
				Platform: specs.Platform{
					OS:   runtime.GOOS,
					Arch: runtime.GOARCH,
				},
				Root: specs.Root{
					Path:     "rootfs",
					Readonly: true,
				},
				Process: specs.Process{
					Terminal: true,
					User:     specs.User{},
					Args: []string{
						"sh",
					},
					Env: []string{
						"PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
						"TERM=xterm",
					},
					Cwd: "/",
				},
				Hostname: "shell",
				Mounts: []specs.MountPoint{
					{
						Name: "proc",
						Path: "/proc",
					},
					{
						Name: "dev",
						Path: "/dev",
					},
					{
						Name: "devpts",
						Path: "/dev/pts",
					},
					{
						Name: "shm",
						Path: "/dev/shm",
					},
					{
						Name: "mqueue",
						Path: "/dev/mqueue",
					},
					{
						Name: "sysfs",
						Path: "/sys",
					},
					{
						Name: "cgroup",
						Path: "/sys/fs/cgroup",
					},
				},
			},
			Linux: specs.Linux{
				Capabilities: []string{
					"CAP_AUDIT_WRITE",
					"CAP_KILL",
					"CAP_NET_BIND_SERVICE",
				},
			},
		}
		rspec := specs.LinuxRuntimeSpec{
			RuntimeSpec: specs.RuntimeSpec{
				Mounts: map[string]specs.Mount{
					"proc": {
						Type:    "proc",
						Source:  "proc",
						Options: nil,
					},
					"dev": {
						Type:    "tmpfs",
						Source:  "tmpfs",
						Options: []string{"nosuid", "strictatime", "mode=755", "size=65536k"},
					},
					"devpts": {
						Type:    "devpts",
						Source:  "devpts",
						Options: []string{"nosuid", "noexec", "newinstance", "ptmxmode=0666", "mode=0620", "gid=5"},
					},
					"shm": {
						Type:    "tmpfs",
						Source:  "shm",
						Options: []string{"nosuid", "noexec", "nodev", "mode=1777", "size=65536k"},
					},
					"mqueue": {
						Type:    "mqueue",
						Source:  "mqueue",
						Options: []string{"nosuid", "noexec", "nodev"},
					},
					"sysfs": {
						Type:    "sysfs",
						Source:  "sysfs",
						Options: []string{"nosuid", "noexec", "nodev"},
					},
					"cgroup": {
						Type:    "cgroup",
						Source:  "cgroup",
						Options: []string{"nosuid", "noexec", "nodev", "relatime", "ro"},
					},
				},
			},
			Linux: specs.LinuxRuntime{
				Namespaces: []specs.Namespace{
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
				Rlimits: []specs.Rlimit{
					{
						Type: "RLIMIT_NOFILE",
						Hard: uint64(1024),
						Soft: uint64(1024),
					},
				},
				Devices: []specs.Device{
					{
						Type:        'c',
						Path:        "/dev/null",
						Major:       1,
						Minor:       3,
						Permissions: "rwm",
						FileMode:    0666,
						UID:         0,
						GID:         0,
					},
					{
						Type:        'c',
						Path:        "/dev/random",
						Major:       1,
						Minor:       8,
						Permissions: "rwm",
						FileMode:    0666,
						UID:         0,
						GID:         0,
					},
					{
						Type:        'c',
						Path:        "/dev/full",
						Major:       1,
						Minor:       7,
						Permissions: "rwm",
						FileMode:    0666,
						UID:         0,
						GID:         0,
					},
					{
						Type:        'c',
						Path:        "/dev/tty",
						Major:       5,
						Minor:       0,
						Permissions: "rwm",
						FileMode:    0666,
						UID:         0,
						GID:         0,
					},
					{
						Type:        'c',
						Path:        "/dev/zero",
						Major:       1,
						Minor:       5,
						Permissions: "rwm",
						FileMode:    0666,
						UID:         0,
						GID:         0,
					},
					{
						Type:        'c',
						Path:        "/dev/urandom",
						Major:       1,
						Minor:       9,
						Permissions: "rwm",
						FileMode:    0666,
						UID:         0,
						GID:         0,
					},
				},
				Resources: &specs.Resources{
					Memory: &specs.Memory{},
				},
				Seccomp: specs.Seccomp{
					DefaultAction: "SCMP_ACT_ALLOW",
					Syscalls:      []*specs.Syscall{},
				},
			},
		}
		checkNoFile := func(name string) error {
			_, err := os.Stat(name)
			if err == nil {
				return fmt.Errorf("File %s exists. Remove it first", name)
			}
			if !os.IsNotExist(err) {
				return err
			}
			return nil
		}
		bundle := context.String("bundle")
		if bundle != "" {
			if err := os.Chdir(bundle); err != nil {
				fatal(err)
			}
		}
		if err := checkNoFile(specConfig); err != nil {
			logrus.Fatal(err)
		}
		if err := checkNoFile(runtimeConfig); err != nil {
			logrus.Fatal(err)
		}
		data, err := json.MarshalIndent(&spec, "", "\t")
		if err != nil {
			logrus.Fatal(err)
		}
		if err := ioutil.WriteFile(specConfig, data, 0666); err != nil {
			logrus.Fatal(err)
		}
		rdata, err := json.MarshalIndent(&rspec, "", "\t")
		if err != nil {
			logrus.Fatal(err)
		}
		if err := ioutil.WriteFile(runtimeConfig, rdata, 0666); err != nil {
			logrus.Fatal(err)
		}
	},
}

var namespaceMapping = map[specs.NamespaceType]configs.NamespaceType{
	specs.PIDNamespace:     configs.NEWPID,
	specs.NetworkNamespace: configs.NEWNET,
	specs.MountNamespace:   configs.NEWNS,
	specs.UserNamespace:    configs.NEWUSER,
	specs.IPCNamespace:     configs.NEWIPC,
	specs.UTSNamespace:     configs.NEWUTS,
}

var mountPropagationMapping = map[string]int{
	"rprivate": syscall.MS_PRIVATE | syscall.MS_REC,
	"private":  syscall.MS_PRIVATE,
	"rslave":   syscall.MS_SLAVE | syscall.MS_REC,
	"slave":    syscall.MS_SLAVE,
	"rshared":  syscall.MS_SHARED | syscall.MS_REC,
	"shared":   syscall.MS_SHARED,
	"":         syscall.MS_PRIVATE | syscall.MS_REC,
}

// validateSpec validates the fields in the spec
// TODO: Add validation for other fields where applicable
func validateSpec(spec *specs.LinuxSpec, rspec *specs.LinuxRuntimeSpec) error {
	if spec.Process.Cwd == "" {
		return fmt.Errorf("Cwd property must not be empty")
	}
	if !filepath.IsAbs(spec.Process.Cwd) {
		return fmt.Errorf("Cwd must be an absolute path")
	}
	return nil
}

// loadSpec loads the specification from the provided path.
// If the path is empty then the default path will be "config.json"
func loadSpec(cPath, rPath string) (spec *specs.LinuxSpec, rspec *specs.LinuxRuntimeSpec, err error) {
	cf, err := os.Open(cPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil, fmt.Errorf("JSON specification file %s not found", cPath)
		}
		return spec, rspec, err
	}
	defer cf.Close()

	rf, err := os.Open(rPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil, fmt.Errorf("JSON runtime config file %s not found", rPath)
		}
		return spec, rspec, err
	}
	defer rf.Close()

	if err = json.NewDecoder(cf).Decode(&spec); err != nil {
		return spec, rspec, err
	}
	if err = json.NewDecoder(rf).Decode(&rspec); err != nil {
		return spec, rspec, err
	}
	if err := checkSpecVersion(spec); err != nil {
		return spec, rspec, err
	}
	return spec, rspec, validateSpec(spec, rspec)
}

// checkSpecVersion makes sure that the spec version matches runc's while we are in the initial
// development period.  It is better to hard fail than have missing fields or options in the spec.
func checkSpecVersion(s *specs.LinuxSpec) error {
	if s.Version != specs.Version {
		return fmt.Errorf("spec version is not compatible with implemented version %q: spec %q", specs.Version, s.Version)
	}
	return nil
}

func createLibcontainerConfig(cgroupName string, spec *specs.LinuxSpec, rspec *specs.LinuxRuntimeSpec) (*configs.Config, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return nil, err
	}
	rootfsPath := spec.Root.Path
	if !filepath.IsAbs(rootfsPath) {
		rootfsPath = filepath.Join(cwd, rootfsPath)
	}
	config := &configs.Config{
		Rootfs:       rootfsPath,
		Capabilities: spec.Linux.Capabilities,
		Readonlyfs:   spec.Root.Readonly,
		Hostname:     spec.Hostname,
	}

	exists := false
	if config.RootPropagation, exists = mountPropagationMapping[rspec.Linux.RootfsPropagation]; !exists {
		return nil, fmt.Errorf("rootfsPropagation=%v is not supported", rspec.Linux.RootfsPropagation)
	}

	for _, ns := range rspec.Linux.Namespaces {
		t, exists := namespaceMapping[ns.Type]
		if !exists {
			return nil, fmt.Errorf("namespace %q does not exist", ns)
		}
		config.Namespaces.Add(t, ns.Path)
	}
	if config.Namespaces.Contains(configs.NEWNET) {
		config.Networks = []*configs.Network{
			{
				Type: "loopback",
			},
		}
	}
	for _, mp := range spec.Mounts {
		m, ok := rspec.Mounts[mp.Name]
		if !ok {
			return nil, fmt.Errorf("Mount with Name %q not found in runtime config", mp.Name)
		}
		config.Mounts = append(config.Mounts, createLibcontainerMount(cwd, mp.Path, m))
	}
	if err := createDevices(rspec, config); err != nil {
		return nil, err
	}
	if err := setupUserNamespace(rspec, config); err != nil {
		return nil, err
	}
	for _, rlimit := range rspec.Linux.Rlimits {
		rl, err := createLibContainerRlimit(rlimit)
		if err != nil {
			return nil, err
		}
		config.Rlimits = append(config.Rlimits, rl)
	}
	c, err := createCgroupConfig(cgroupName, rspec, config.Devices)
	if err != nil {
		return nil, err
	}
	config.Cgroups = c
	if config.Readonlyfs {
		setReadonly(config)
		config.MaskPaths = []string{
			"/proc/kcore",
		}
		config.ReadonlyPaths = []string{
			"/proc/sys", "/proc/sysrq-trigger", "/proc/irq", "/proc/bus",
		}
	}
	seccomp, err := setupSeccomp(&rspec.Linux.Seccomp)
	if err != nil {
		return nil, err
	}
	config.Seccomp = seccomp
	config.Sysctl = rspec.Linux.Sysctl
	config.ProcessLabel = rspec.Linux.SelinuxProcessLabel
	config.AppArmorProfile = rspec.Linux.ApparmorProfile
	for _, g := range spec.Process.User.AdditionalGids {
		config.AdditionalGroups = append(config.AdditionalGroups, strconv.FormatUint(uint64(g), 10))
	}
	createHooks(rspec, config)
	config.Version = specs.Version
	return config, nil
}

func createLibcontainerMount(cwd, dest string, m specs.Mount) *configs.Mount {
	flags, pgflags, data := parseMountOptions(m.Options)
	source := m.Source
	if m.Type == "bind" {
		if !filepath.IsAbs(source) {
			source = filepath.Join(cwd, m.Source)
		}
	}
	return &configs.Mount{
		Device:           m.Type,
		Source:           source,
		Destination:      dest,
		Data:             data,
		Flags:            flags,
		PropagationFlags: pgflags,
	}
}

func createCgroupConfig(name string, spec *specs.LinuxRuntimeSpec, devices []*configs.Device) (*configs.Cgroup, error) {
	myCgroupPath, err := cgroups.GetThisCgroupDir("devices")
	if err != nil {
		return nil, err
	}
	c := &configs.Cgroup{
		Name:      name,
		Parent:    myCgroupPath,
		Resources: &configs.Resources{},
	}
	c.Resources.AllowedDevices = append(devices, allowedDevices...)
	r := spec.Linux.Resources
	if r != nil {
		if r.Memory != nil {
			if r.Memory.Limit != nil {
				c.Resources.Memory = int64(*r.Memory.Limit)
			}
			if r.Memory.Reservation != nil {
				c.Resources.MemoryReservation = int64(*r.Memory.Reservation)
			}
			if r.Memory.Swap != nil {
				c.Resources.MemorySwap = int64(*r.Memory.Swap)
			}
			if r.Memory.Kernel != nil {
				c.Resources.KernelMemory = int64(*r.Memory.Kernel)
			}
			if r.Memory.Swappiness != nil {
				c.Resources.MemorySwappiness = int64(*r.Memory.Swappiness)
			}
		}

		if r.CPU != nil {
			if r.CPU.Shares != nil {
				c.Resources.CpuShares = int64(*r.CPU.Shares)
			}
			if r.CPU.Quota != nil {
				c.Resources.CpuQuota = int64(*r.CPU.Quota)
			}
			if r.CPU.Period != nil {
				c.Resources.CpuPeriod = int64(*r.CPU.Period)
			}
			if r.CPU.RealtimeRuntime != nil {
				c.Resources.CpuRtRuntime = int64(*r.CPU.RealtimeRuntime)
			}
			if r.CPU.RealtimePeriod != nil {
				c.Resources.CpuRtPeriod = int64(*r.CPU.RealtimePeriod)
			}
			if r.CPU.Cpus != nil {
				c.Resources.CpusetCpus = *r.CPU.Cpus
			}
			if r.CPU.Mems != nil {
				c.Resources.CpusetMems = *r.CPU.Mems
			}
		}
		if r.Pids != nil {
			c.Resources.PidsLimit = *r.Pids.Limit
		}
		if r.BlockIO != nil {
			if r.BlockIO.Weight != nil {
				c.Resources.BlkioWeight = *r.BlockIO.Weight
			}
			if r.BlockIO.LeafWeight != nil {
				c.Resources.BlkioLeafWeight = *r.BlockIO.LeafWeight
			}
			if r.BlockIO.WeightDevice != nil {
				for _, wd := range r.BlockIO.WeightDevice {
					weightDevice := configs.NewWeightDevice(wd.Major, wd.Minor, *wd.Weight, *wd.LeafWeight)
					c.Resources.BlkioWeightDevice = append(c.Resources.BlkioWeightDevice, weightDevice)
				}
			}
			if r.BlockIO.ThrottleReadBpsDevice != nil {
				for _, td := range r.BlockIO.ThrottleReadBpsDevice {
					throttleDevice := configs.NewThrottleDevice(td.Major, td.Minor, *td.Rate)
					c.Resources.BlkioThrottleReadBpsDevice = append(c.Resources.BlkioThrottleReadBpsDevice, throttleDevice)
				}
			}
			if r.BlockIO.ThrottleWriteBpsDevice != nil {
				for _, td := range r.BlockIO.ThrottleWriteBpsDevice {
					throttleDevice := configs.NewThrottleDevice(td.Major, td.Minor, *td.Rate)
					c.Resources.BlkioThrottleWriteBpsDevice = append(c.Resources.BlkioThrottleWriteBpsDevice, throttleDevice)
				}
			}
			if r.BlockIO.ThrottleReadIOPSDevice != nil {
				for _, td := range r.BlockIO.ThrottleReadIOPSDevice {
					throttleDevice := configs.NewThrottleDevice(td.Major, td.Minor, *td.Rate)
					c.Resources.BlkioThrottleReadIOPSDevice = append(c.Resources.BlkioThrottleReadIOPSDevice, throttleDevice)
				}
			}
			if r.BlockIO.ThrottleWriteIOPSDevice != nil {
				for _, td := range r.BlockIO.ThrottleWriteIOPSDevice {
					throttleDevice := configs.NewThrottleDevice(td.Major, td.Minor, *td.Rate)
					c.Resources.BlkioThrottleWriteIOPSDevice = append(c.Resources.BlkioThrottleWriteIOPSDevice, throttleDevice)
				}
			}
		}
		for _, l := range r.HugepageLimits {
			c.Resources.HugetlbLimit = append(c.Resources.HugetlbLimit, &configs.HugepageLimit{
				Pagesize: *l.Pagesize,
				Limit:    *l.Limit,
			})
		}
		if r.DisableOOMKiller != nil {
			c.Resources.OomKillDisable = *r.DisableOOMKiller
		}
		if r.Network != nil {
			if r.Network.ClassID != nil {
				c.Resources.NetClsClassid = string(*r.Network.ClassID)
			}
			for _, m := range r.Network.Priorities {
				c.Resources.NetPrioIfpriomap = append(c.Resources.NetPrioIfpriomap, &configs.IfPrioMap{
					Interface: m.Name,
					Priority:  int64(m.Priority),
				})
			}
		}
	}
	return c, nil
}

func createDevices(spec *specs.LinuxRuntimeSpec, config *configs.Config) error {
	for _, d := range spec.Linux.Devices {
		device := &configs.Device{
			Type:        d.Type,
			Path:        d.Path,
			Major:       d.Major,
			Minor:       d.Minor,
			Permissions: d.Permissions,
			FileMode:    d.FileMode,
			Uid:         d.UID,
			Gid:         d.GID,
		}
		config.Devices = append(config.Devices, device)
	}
	return nil
}

func setReadonly(config *configs.Config) {
	for _, m := range config.Mounts {
		if m.Device == "sysfs" {
			m.Flags |= syscall.MS_RDONLY
		}
	}
}

func setupUserNamespace(spec *specs.LinuxRuntimeSpec, config *configs.Config) error {
	if len(spec.Linux.UIDMappings) == 0 {
		return nil
	}
	config.Namespaces.Add(configs.NEWUSER, "")
	create := func(m specs.IDMapping) configs.IDMap {
		return configs.IDMap{
			HostID:      int(m.HostID),
			ContainerID: int(m.ContainerID),
			Size:        int(m.Size),
		}
	}
	for _, m := range spec.Linux.UIDMappings {
		config.UidMappings = append(config.UidMappings, create(m))
	}
	for _, m := range spec.Linux.GIDMappings {
		config.GidMappings = append(config.GidMappings, create(m))
	}
	rootUID, err := config.HostUID()
	if err != nil {
		return err
	}
	rootGID, err := config.HostGID()
	if err != nil {
		return err
	}
	for _, node := range config.Devices {
		node.Uid = uint32(rootUID)
		node.Gid = uint32(rootGID)
	}
	return nil
}

func createLibContainerRlimit(rlimit specs.Rlimit) (configs.Rlimit, error) {
	rl, err := strToRlimit(rlimit.Type)
	if err != nil {
		return configs.Rlimit{}, err
	}
	return configs.Rlimit{
		Type: rl,
		Hard: uint64(rlimit.Hard),
		Soft: uint64(rlimit.Soft),
	}, nil
}

// parseMountOptions parses the string and returns the flags, propagation
// flags and any mount data that it contains.
func parseMountOptions(options []string) (int, []int, string) {
	var (
		flag   int
		pgflag []int
		data   []string
	)
	flags := map[string]struct {
		clear bool
		flag  int
	}{
		"async":         {true, syscall.MS_SYNCHRONOUS},
		"atime":         {true, syscall.MS_NOATIME},
		"bind":          {false, syscall.MS_BIND},
		"defaults":      {false, 0},
		"dev":           {true, syscall.MS_NODEV},
		"diratime":      {true, syscall.MS_NODIRATIME},
		"dirsync":       {false, syscall.MS_DIRSYNC},
		"exec":          {true, syscall.MS_NOEXEC},
		"mand":          {false, syscall.MS_MANDLOCK},
		"noatime":       {false, syscall.MS_NOATIME},
		"nodev":         {false, syscall.MS_NODEV},
		"nodiratime":    {false, syscall.MS_NODIRATIME},
		"noexec":        {false, syscall.MS_NOEXEC},
		"nomand":        {true, syscall.MS_MANDLOCK},
		"norelatime":    {true, syscall.MS_RELATIME},
		"nostrictatime": {true, syscall.MS_STRICTATIME},
		"nosuid":        {false, syscall.MS_NOSUID},
		"rbind":         {false, syscall.MS_BIND | syscall.MS_REC},
		"relatime":      {false, syscall.MS_RELATIME},
		"remount":       {false, syscall.MS_REMOUNT},
		"ro":            {false, syscall.MS_RDONLY},
		"rw":            {true, syscall.MS_RDONLY},
		"strictatime":   {false, syscall.MS_STRICTATIME},
		"suid":          {true, syscall.MS_NOSUID},
		"sync":          {false, syscall.MS_SYNCHRONOUS},
	}
	propagationFlags := map[string]struct {
		clear bool
		flag  int
	}{
		"private":     {false, syscall.MS_PRIVATE},
		"shared":      {false, syscall.MS_SHARED},
		"slave":       {false, syscall.MS_SLAVE},
		"unbindable":  {false, syscall.MS_UNBINDABLE},
		"rprivate":    {false, syscall.MS_PRIVATE | syscall.MS_REC},
		"rshared":     {false, syscall.MS_SHARED | syscall.MS_REC},
		"rslave":      {false, syscall.MS_SLAVE | syscall.MS_REC},
		"runbindable": {false, syscall.MS_UNBINDABLE | syscall.MS_REC},
	}
	for _, o := range options {
		// If the option does not exist in the flags table or the flag
		// is not supported on the platform,
		// then it is a data value for a specific fs type
		if f, exists := flags[o]; exists && f.flag != 0 {
			if f.clear {
				flag &= ^f.flag
			} else {
				flag |= f.flag
			}
		} else if f, exists := propagationFlags[o]; exists && f.flag != 0 {
			pgflag = append(pgflag, f.flag)
		} else {
			data = append(data, o)
		}
	}
	return flag, pgflag, strings.Join(data, ",")
}

func setupSeccomp(config *specs.Seccomp) (*configs.Seccomp, error) {
	if config == nil {
		return nil, nil
	}

	// No default action specified, no syscalls listed, assume seccomp disabled
	if config.DefaultAction == "" && len(config.Syscalls) == 0 {
		return nil, nil
	}

	newConfig := new(configs.Seccomp)
	newConfig.Syscalls = []*configs.Syscall{}

	if len(config.Architectures) > 0 {
		newConfig.Architectures = []string{}
		for _, arch := range config.Architectures {
			newArch, err := seccomp.ConvertStringToArch(string(arch))
			if err != nil {
				return nil, err
			}
			newConfig.Architectures = append(newConfig.Architectures, newArch)
		}
	}

	// Convert default action from string representation
	newDefaultAction, err := seccomp.ConvertStringToAction(string(config.DefaultAction))
	if err != nil {
		return nil, err
	}
	newConfig.DefaultAction = newDefaultAction

	// Loop through all syscall blocks and convert them to libcontainer format
	for _, call := range config.Syscalls {
		newAction, err := seccomp.ConvertStringToAction(string(call.Action))
		if err != nil {
			return nil, err
		}

		newCall := configs.Syscall{
			Name:   call.Name,
			Action: newAction,
			Args:   []*configs.Arg{},
		}

		// Loop through all the arguments of the syscall and convert them
		for _, arg := range call.Args {
			newOp, err := seccomp.ConvertStringToOperator(string(arg.Op))
			if err != nil {
				return nil, err
			}

			newArg := configs.Arg{
				Index:    arg.Index,
				Value:    arg.Value,
				ValueTwo: arg.ValueTwo,
				Op:       newOp,
			}

			newCall.Args = append(newCall.Args, &newArg)
		}

		newConfig.Syscalls = append(newConfig.Syscalls, &newCall)
	}

	return newConfig, nil
}

func createHooks(rspec *specs.LinuxRuntimeSpec, config *configs.Config) {
	config.Hooks = &configs.Hooks{}
	for _, h := range rspec.Hooks.Prestart {
		cmd := configs.Command{
			Path: h.Path,
			Args: h.Args,
			Env:  h.Env,
		}
		config.Hooks.Prestart = append(config.Hooks.Prestart, configs.NewCommandHook(cmd))
	}
	for _, h := range rspec.Hooks.Poststart {
		cmd := configs.Command{
			Path: h.Path,
			Args: h.Args,
			Env:  h.Env,
		}
		config.Hooks.Poststart = append(config.Hooks.Poststart, configs.NewCommandHook(cmd))
	}
	for _, h := range rspec.Hooks.Poststop {
		cmd := configs.Command{
			Path: h.Path,
			Args: h.Args,
			Env:  h.Env,
		}
		config.Hooks.Poststop = append(config.Hooks.Poststop, configs.NewCommandHook(cmd))
	}
}
