// +build linux

// Package specconv implements conversion of specifications to libcontainer
// configurations
package specconv

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runc/libcontainer/seccomp"
	libcontainerUtils "github.com/opencontainers/runc/libcontainer/utils"
	"github.com/opencontainers/runtime-spec/specs-go"

	"golang.org/x/sys/unix"
)

const wildcard = -1

var namespaceMapping = map[specs.LinuxNamespaceType]configs.NamespaceType{
	specs.PIDNamespace:     configs.NEWPID,
	specs.NetworkNamespace: configs.NEWNET,
	specs.MountNamespace:   configs.NEWNS,
	specs.UserNamespace:    configs.NEWUSER,
	specs.IPCNamespace:     configs.NEWIPC,
	specs.UTSNamespace:     configs.NEWUTS,
}

var mountPropagationMapping = map[string]int{
	"rprivate": unix.MS_PRIVATE | unix.MS_REC,
	"private":  unix.MS_PRIVATE,
	"rslave":   unix.MS_SLAVE | unix.MS_REC,
	"slave":    unix.MS_SLAVE,
	"rshared":  unix.MS_SHARED | unix.MS_REC,
	"shared":   unix.MS_SHARED,
	"":         unix.MS_PRIVATE | unix.MS_REC,
}

var allowedDevices = []*configs.Device{
	// allow mknod for any device
	{
		Type:        'c',
		Major:       wildcard,
		Minor:       wildcard,
		Permissions: "m",
		Allow:       true,
	},
	{
		Type:        'b',
		Major:       wildcard,
		Minor:       wildcard,
		Permissions: "m",
		Allow:       true,
	},
	{
		Type:        'c',
		Path:        "/dev/null",
		Major:       1,
		Minor:       3,
		Permissions: "rwm",
		Allow:       true,
	},
	{
		Type:        'c',
		Path:        "/dev/random",
		Major:       1,
		Minor:       8,
		Permissions: "rwm",
		Allow:       true,
	},
	{
		Type:        'c',
		Path:        "/dev/full",
		Major:       1,
		Minor:       7,
		Permissions: "rwm",
		Allow:       true,
	},
	{
		Type:        'c',
		Path:        "/dev/tty",
		Major:       5,
		Minor:       0,
		Permissions: "rwm",
		Allow:       true,
	},
	{
		Type:        'c',
		Path:        "/dev/zero",
		Major:       1,
		Minor:       5,
		Permissions: "rwm",
		Allow:       true,
	},
	{
		Type:        'c',
		Path:        "/dev/urandom",
		Major:       1,
		Minor:       9,
		Permissions: "rwm",
		Allow:       true,
	},
	{
		Path:        "/dev/console",
		Type:        'c',
		Major:       5,
		Minor:       1,
		Permissions: "rwm",
		Allow:       true,
	},
	// /dev/pts/ - pts namespaces are "coming soon"
	{
		Path:        "",
		Type:        'c',
		Major:       136,
		Minor:       wildcard,
		Permissions: "rwm",
		Allow:       true,
	},
	{
		Path:        "",
		Type:        'c',
		Major:       5,
		Minor:       2,
		Permissions: "rwm",
		Allow:       true,
	},
	// tuntap
	{
		Path:        "",
		Type:        'c',
		Major:       10,
		Minor:       200,
		Permissions: "rwm",
		Allow:       true,
	},
}

type CreateOpts struct {
	CgroupName       string
	UseSystemdCgroup bool
	NoPivotRoot      bool
	NoNewKeyring     bool
	Spec             *specs.Spec
	Rootless         bool
}

// CreateLibcontainerConfig creates a new libcontainer configuration from a
// given specification and a cgroup name
func CreateLibcontainerConfig(opts *CreateOpts) (*configs.Config, error) {
	// runc's cwd will always be the bundle path
	rcwd, err := os.Getwd()
	if err != nil {
		return nil, err
	}
	cwd, err := filepath.Abs(rcwd)
	if err != nil {
		return nil, err
	}
	spec := opts.Spec
	if spec.Root == nil {
		return nil, fmt.Errorf("Root must be specified")
	}
	rootfsPath := spec.Root.Path
	if !filepath.IsAbs(rootfsPath) {
		rootfsPath = filepath.Join(cwd, rootfsPath)
	}
	labels := []string{}
	for k, v := range spec.Annotations {
		labels = append(labels, fmt.Sprintf("%s=%s", k, v))
	}
	config := &configs.Config{
		Rootfs:       rootfsPath,
		NoPivotRoot:  opts.NoPivotRoot,
		Readonlyfs:   spec.Root.Readonly,
		Hostname:     spec.Hostname,
		Labels:       append(labels, fmt.Sprintf("bundle=%s", cwd)),
		NoNewKeyring: opts.NoNewKeyring,
		Rootless:     opts.Rootless,
	}

	exists := false
	if config.Namespaces.Contains(configs.NEWNET) {
		config.Networks = []*configs.Network{
			{
				Type: "loopback",
			},
		}
	}
	for _, m := range spec.Mounts {
		config.Mounts = append(config.Mounts, createLibcontainerMount(cwd, m))
	}
	if err := createDevices(spec, config); err != nil {
		return nil, err
	}
	if err := setupUserNamespace(spec, config); err != nil {
		return nil, err
	}
	c, err := createCgroupConfig(opts)
	if err != nil {
		return nil, err
	}
	config.Cgroups = c
	// set linux-specific config
	if spec.Linux != nil {
		if config.RootPropagation, exists = mountPropagationMapping[spec.Linux.RootfsPropagation]; !exists {
			return nil, fmt.Errorf("rootfsPropagation=%v is not supported", spec.Linux.RootfsPropagation)
		}

		for _, ns := range spec.Linux.Namespaces {
			t, exists := namespaceMapping[ns.Type]
			if !exists {
				return nil, fmt.Errorf("namespace %q does not exist", ns)
			}
			if config.Namespaces.Contains(t) {
				return nil, fmt.Errorf("malformed spec file: duplicated ns %q", ns)
			}
			config.Namespaces.Add(t, ns.Path)
		}
		config.MaskPaths = spec.Linux.MaskedPaths
		config.ReadonlyPaths = spec.Linux.ReadonlyPaths
		config.MountLabel = spec.Linux.MountLabel
		config.Sysctl = spec.Linux.Sysctl
		if spec.Linux.Seccomp != nil {
			seccomp, err := setupSeccomp(spec.Linux.Seccomp)
			if err != nil {
				return nil, err
			}
			config.Seccomp = seccomp
		}
	}
	if spec.Process.SelinuxLabel != "" {
		config.ProcessLabel = spec.Process.SelinuxLabel
	}
	if spec.Process != nil && spec.Process.OOMScoreAdj != nil {
		config.OomScoreAdj = *spec.Process.OOMScoreAdj
	}
	if spec.Process.Capabilities != nil {
		config.Capabilities = &configs.Capabilities{
			Bounding:    spec.Process.Capabilities.Bounding,
			Effective:   spec.Process.Capabilities.Effective,
			Permitted:   spec.Process.Capabilities.Permitted,
			Inheritable: spec.Process.Capabilities.Inheritable,
			Ambient:     spec.Process.Capabilities.Ambient,
		}
	}
	createHooks(spec, config)
	config.Version = specs.Version
	return config, nil
}

func createLibcontainerMount(cwd string, m specs.Mount) *configs.Mount {
	flags, pgflags, data, ext := parseMountOptions(m.Options)
	source := m.Source
	if m.Type == "bind" {
		if !filepath.IsAbs(source) {
			source = filepath.Join(cwd, m.Source)
		}
	}
	return &configs.Mount{
		Device:           m.Type,
		Source:           source,
		Destination:      m.Destination,
		Data:             data,
		Flags:            flags,
		PropagationFlags: pgflags,
		Extensions:       ext,
	}
}

func createCgroupConfig(opts *CreateOpts) (*configs.Cgroup, error) {
	var (
		myCgroupPath string

		spec             = opts.Spec
		useSystemdCgroup = opts.UseSystemdCgroup
		name             = opts.CgroupName
	)

	c := &configs.Cgroup{
		Resources: &configs.Resources{},
	}

	if spec.Linux != nil && spec.Linux.CgroupsPath != "" {
		myCgroupPath = libcontainerUtils.CleanPath(spec.Linux.CgroupsPath)
		if useSystemdCgroup {
			myCgroupPath = spec.Linux.CgroupsPath
		}
	}

	if useSystemdCgroup {
		if myCgroupPath == "" {
			c.Parent = "system.slice"
			c.ScopePrefix = "runc"
			c.Name = name
		} else {
			// Parse the path from expected "slice:prefix:name"
			// for e.g. "system.slice:docker:1234"
			parts := strings.Split(myCgroupPath, ":")
			if len(parts) != 3 {
				return nil, fmt.Errorf("expected cgroupsPath to be of format \"slice:prefix:name\" for systemd cgroups")
			}
			c.Parent = parts[0]
			c.ScopePrefix = parts[1]
			c.Name = parts[2]
		}
	} else {
		if myCgroupPath == "" {
			c.Name = name
		}
		c.Path = myCgroupPath
	}

	// In rootless containers, any attempt to make cgroup changes will fail.
	// libcontainer will validate this and we shouldn't add any cgroup options
	// the user didn't specify.
	if !opts.Rootless {
		c.Resources.AllowedDevices = allowedDevices
	}
	if spec.Linux != nil {
		r := spec.Linux.Resources
		if r == nil {
			return c, nil
		}
		for i, d := range spec.Linux.Resources.Devices {
			var (
				t     = "a"
				major = int64(-1)
				minor = int64(-1)
			)
			if d.Type != "" {
				t = d.Type
			}
			if d.Major != nil {
				major = *d.Major
			}
			if d.Minor != nil {
				minor = *d.Minor
			}
			if d.Access == "" {
				return nil, fmt.Errorf("device access at %d field cannot be empty", i)
			}
			dt, err := stringToCgroupDeviceRune(t)
			if err != nil {
				return nil, err
			}
			dd := &configs.Device{
				Type:        dt,
				Major:       major,
				Minor:       minor,
				Permissions: d.Access,
				Allow:       d.Allow,
			}
			c.Resources.Devices = append(c.Resources.Devices, dd)
		}
		if r.Memory != nil {
			if r.Memory.Limit != nil {
				c.Resources.Memory = *r.Memory.Limit
			}
			if r.Memory.Reservation != nil {
				c.Resources.MemoryReservation = *r.Memory.Reservation
			}
			if r.Memory.Swap != nil {
				c.Resources.MemorySwap = *r.Memory.Swap
			}
			if r.Memory.Kernel != nil {
				c.Resources.KernelMemory = *r.Memory.Kernel
			}
			if r.Memory.KernelTCP != nil {
				c.Resources.KernelMemoryTCP = *r.Memory.KernelTCP
			}
			if r.Memory.Swappiness != nil {
				c.Resources.MemorySwappiness = r.Memory.Swappiness
			}
			if r.Memory.DisableOOMKiller != nil {
				c.Resources.OomKillDisable = *r.Memory.DisableOOMKiller
			}
		}
		if r.CPU != nil {
			if r.CPU.Shares != nil {
				c.Resources.CpuShares = *r.CPU.Shares
			}
			if r.CPU.Quota != nil {
				c.Resources.CpuQuota = *r.CPU.Quota
			}
			if r.CPU.Period != nil {
				c.Resources.CpuPeriod = *r.CPU.Period
			}
			if r.CPU.RealtimeRuntime != nil {
				c.Resources.CpuRtRuntime = *r.CPU.RealtimeRuntime
			}
			if r.CPU.RealtimePeriod != nil {
				c.Resources.CpuRtPeriod = *r.CPU.RealtimePeriod
			}
			if r.CPU.Cpus != "" {
				c.Resources.CpusetCpus = r.CPU.Cpus
			}
			if r.CPU.Mems != "" {
				c.Resources.CpusetMems = r.CPU.Mems
			}
		}
		if r.Pids != nil {
			c.Resources.PidsLimit = r.Pids.Limit
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
					var weight, leafWeight uint16
					if wd.Weight != nil {
						weight = *wd.Weight
					}
					if wd.LeafWeight != nil {
						leafWeight = *wd.LeafWeight
					}
					weightDevice := configs.NewWeightDevice(wd.Major, wd.Minor, weight, leafWeight)
					c.Resources.BlkioWeightDevice = append(c.Resources.BlkioWeightDevice, weightDevice)
				}
			}
			if r.BlockIO.ThrottleReadBpsDevice != nil {
				for _, td := range r.BlockIO.ThrottleReadBpsDevice {
					rate := td.Rate
					throttleDevice := configs.NewThrottleDevice(td.Major, td.Minor, rate)
					c.Resources.BlkioThrottleReadBpsDevice = append(c.Resources.BlkioThrottleReadBpsDevice, throttleDevice)
				}
			}
			if r.BlockIO.ThrottleWriteBpsDevice != nil {
				for _, td := range r.BlockIO.ThrottleWriteBpsDevice {
					rate := td.Rate
					throttleDevice := configs.NewThrottleDevice(td.Major, td.Minor, rate)
					c.Resources.BlkioThrottleWriteBpsDevice = append(c.Resources.BlkioThrottleWriteBpsDevice, throttleDevice)
				}
			}
			if r.BlockIO.ThrottleReadIOPSDevice != nil {
				for _, td := range r.BlockIO.ThrottleReadIOPSDevice {
					rate := td.Rate
					throttleDevice := configs.NewThrottleDevice(td.Major, td.Minor, rate)
					c.Resources.BlkioThrottleReadIOPSDevice = append(c.Resources.BlkioThrottleReadIOPSDevice, throttleDevice)
				}
			}
			if r.BlockIO.ThrottleWriteIOPSDevice != nil {
				for _, td := range r.BlockIO.ThrottleWriteIOPSDevice {
					rate := td.Rate
					throttleDevice := configs.NewThrottleDevice(td.Major, td.Minor, rate)
					c.Resources.BlkioThrottleWriteIOPSDevice = append(c.Resources.BlkioThrottleWriteIOPSDevice, throttleDevice)
				}
			}
		}
		for _, l := range r.HugepageLimits {
			c.Resources.HugetlbLimit = append(c.Resources.HugetlbLimit, &configs.HugepageLimit{
				Pagesize: l.Pagesize,
				Limit:    l.Limit,
			})
		}
		if r.Network != nil {
			if r.Network.ClassID != nil {
				c.Resources.NetClsClassid = *r.Network.ClassID
			}
			for _, m := range r.Network.Priorities {
				c.Resources.NetPrioIfpriomap = append(c.Resources.NetPrioIfpriomap, &configs.IfPrioMap{
					Interface: m.Name,
					Priority:  int64(m.Priority),
				})
			}
		}
	}
	if !opts.Rootless {
		// append the default allowed devices to the end of the list
		c.Resources.Devices = append(c.Resources.Devices, allowedDevices...)
	}
	return c, nil
}

func stringToCgroupDeviceRune(s string) (rune, error) {
	switch s {
	case "a":
		return 'a', nil
	case "b":
		return 'b', nil
	case "c":
		return 'c', nil
	default:
		return 0, fmt.Errorf("invalid cgroup device type %q", s)
	}
}

func stringToDeviceRune(s string) (rune, error) {
	switch s {
	case "p":
		return 'p', nil
	case "u":
		return 'u', nil
	case "b":
		return 'b', nil
	case "c":
		return 'c', nil
	default:
		return 0, fmt.Errorf("invalid device type %q", s)
	}
}

func createDevices(spec *specs.Spec, config *configs.Config) error {
	// add whitelisted devices
	config.Devices = []*configs.Device{
		{
			Type:     'c',
			Path:     "/dev/null",
			Major:    1,
			Minor:    3,
			FileMode: 0666,
			Uid:      0,
			Gid:      0,
		},
		{
			Type:     'c',
			Path:     "/dev/random",
			Major:    1,
			Minor:    8,
			FileMode: 0666,
			Uid:      0,
			Gid:      0,
		},
		{
			Type:     'c',
			Path:     "/dev/full",
			Major:    1,
			Minor:    7,
			FileMode: 0666,
			Uid:      0,
			Gid:      0,
		},
		{
			Type:     'c',
			Path:     "/dev/tty",
			Major:    5,
			Minor:    0,
			FileMode: 0666,
			Uid:      0,
			Gid:      0,
		},
		{
			Type:     'c',
			Path:     "/dev/zero",
			Major:    1,
			Minor:    5,
			FileMode: 0666,
			Uid:      0,
			Gid:      0,
		},
		{
			Type:     'c',
			Path:     "/dev/urandom",
			Major:    1,
			Minor:    9,
			FileMode: 0666,
			Uid:      0,
			Gid:      0,
		},
	}
	// merge in additional devices from the spec
	if spec.Linux != nil {
		for _, d := range spec.Linux.Devices {
			var uid, gid uint32
			var filemode os.FileMode = 0666

			if d.UID != nil {
				uid = *d.UID
			}
			if d.GID != nil {
				gid = *d.GID
			}
			dt, err := stringToDeviceRune(d.Type)
			if err != nil {
				return err
			}
			if d.FileMode != nil {
				filemode = *d.FileMode
			}
			device := &configs.Device{
				Type:     dt,
				Path:     d.Path,
				Major:    d.Major,
				Minor:    d.Minor,
				FileMode: filemode,
				Uid:      uid,
				Gid:      gid,
			}
			config.Devices = append(config.Devices, device)
		}
	}
	return nil
}

func setupUserNamespace(spec *specs.Spec, config *configs.Config) error {
	create := func(m specs.LinuxIDMapping) configs.IDMap {
		return configs.IDMap{
			HostID:      int(m.HostID),
			ContainerID: int(m.ContainerID),
			Size:        int(m.Size),
		}
	}
	if spec.Linux != nil {
		if len(spec.Linux.UIDMappings) == 0 {
			return nil
		}
		for _, m := range spec.Linux.UIDMappings {
			config.UidMappings = append(config.UidMappings, create(m))
		}
		for _, m := range spec.Linux.GIDMappings {
			config.GidMappings = append(config.GidMappings, create(m))
		}
	}
	rootUID, err := config.HostRootUID()
	if err != nil {
		return err
	}
	rootGID, err := config.HostRootGID()
	if err != nil {
		return err
	}
	for _, node := range config.Devices {
		node.Uid = uint32(rootUID)
		node.Gid = uint32(rootGID)
	}
	return nil
}

// parseMountOptions parses the string and returns the flags, propagation
// flags and any mount data that it contains.
func parseMountOptions(options []string) (int, []int, string, int) {
	var (
		flag     int
		pgflag   []int
		data     []string
		extFlags int
	)
	flags := map[string]struct {
		clear bool
		flag  int
	}{
		"acl":           {false, unix.MS_POSIXACL},
		"async":         {true, unix.MS_SYNCHRONOUS},
		"atime":         {true, unix.MS_NOATIME},
		"bind":          {false, unix.MS_BIND},
		"defaults":      {false, 0},
		"dev":           {true, unix.MS_NODEV},
		"diratime":      {true, unix.MS_NODIRATIME},
		"dirsync":       {false, unix.MS_DIRSYNC},
		"exec":          {true, unix.MS_NOEXEC},
		"iversion":      {false, unix.MS_I_VERSION},
		"lazytime":      {false, unix.MS_LAZYTIME},
		"loud":          {true, unix.MS_SILENT},
		"mand":          {false, unix.MS_MANDLOCK},
		"noacl":         {true, unix.MS_POSIXACL},
		"noatime":       {false, unix.MS_NOATIME},
		"nodev":         {false, unix.MS_NODEV},
		"nodiratime":    {false, unix.MS_NODIRATIME},
		"noexec":        {false, unix.MS_NOEXEC},
		"noiversion":    {true, unix.MS_I_VERSION},
		"nolazytime":    {true, unix.MS_LAZYTIME},
		"nomand":        {true, unix.MS_MANDLOCK},
		"norelatime":    {true, unix.MS_RELATIME},
		"nostrictatime": {true, unix.MS_STRICTATIME},
		"nosuid":        {false, unix.MS_NOSUID},
		"rbind":         {false, unix.MS_BIND | unix.MS_REC},
		"relatime":      {false, unix.MS_RELATIME},
		"remount":       {false, unix.MS_REMOUNT},
		"ro":            {false, unix.MS_RDONLY},
		"rw":            {true, unix.MS_RDONLY},
		"silent":        {false, unix.MS_SILENT},
		"strictatime":   {false, unix.MS_STRICTATIME},
		"suid":          {true, unix.MS_NOSUID},
		"sync":          {false, unix.MS_SYNCHRONOUS},
	}
	propagationFlags := map[string]int{
		"private":     unix.MS_PRIVATE,
		"shared":      unix.MS_SHARED,
		"slave":       unix.MS_SLAVE,
		"unbindable":  unix.MS_UNBINDABLE,
		"rprivate":    unix.MS_PRIVATE | unix.MS_REC,
		"rshared":     unix.MS_SHARED | unix.MS_REC,
		"rslave":      unix.MS_SLAVE | unix.MS_REC,
		"runbindable": unix.MS_UNBINDABLE | unix.MS_REC,
	}
	extensionFlags := map[string]struct {
		clear bool
		flag  int
	}{
		"tmpcopyup": {false, configs.EXT_COPYUP},
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
		} else if f, exists := propagationFlags[o]; exists && f != 0 {
			pgflag = append(pgflag, f)
		} else if f, exists := extensionFlags[o]; exists && f.flag != 0 {
			if f.clear {
				extFlags &= ^f.flag
			} else {
				extFlags |= f.flag
			}
		} else {
			data = append(data, o)
		}
	}
	return flag, pgflag, strings.Join(data, ","), extFlags
}

func setupSeccomp(config *specs.LinuxSeccomp) (*configs.Seccomp, error) {
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

		for _, name := range call.Names {
			newCall := configs.Syscall{
				Name:   name,
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
	}

	return newConfig, nil
}

func createHooks(rspec *specs.Spec, config *configs.Config) {
	config.Hooks = &configs.Hooks{}
	if rspec.Hooks != nil {

		for _, h := range rspec.Hooks.Prestart {
			cmd := createCommandHook(h)
			config.Hooks.Prestart = append(config.Hooks.Prestart, configs.NewCommandHook(cmd))
		}
		for _, h := range rspec.Hooks.Poststart {
			cmd := createCommandHook(h)
			config.Hooks.Poststart = append(config.Hooks.Poststart, configs.NewCommandHook(cmd))
		}
		for _, h := range rspec.Hooks.Poststop {
			cmd := createCommandHook(h)
			config.Hooks.Poststop = append(config.Hooks.Poststop, configs.NewCommandHook(cmd))
		}
	}
}

func createCommandHook(h specs.Hook) configs.Command {
	cmd := configs.Command{
		Path: h.Path,
		Args: h.Args,
		Env:  h.Env,
	}
	if h.Timeout != nil {
		d := time.Duration(*h.Timeout) * time.Second
		cmd.Timeout = &d
	}
	return cmd
}
