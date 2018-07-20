package daemon

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	containertypes "github.com/docker/docker/api/types/container"
	"github.com/docker/docker/container"
	"github.com/docker/docker/daemon/caps"
	daemonconfig "github.com/docker/docker/daemon/config"
	"github.com/docker/docker/oci"
	"github.com/docker/docker/pkg/idtools"
	"github.com/docker/docker/pkg/mount"
	"github.com/docker/docker/volume"
	"github.com/opencontainers/runc/libcontainer/apparmor"
	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/devices"
	"github.com/opencontainers/runc/libcontainer/user"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

// nolint: gosimple
var (
	deviceCgroupRuleRegex = regexp.MustCompile("^([acb]) ([0-9]+|\\*):([0-9]+|\\*) ([rwm]{1,3})$")
)

func setResources(s *specs.Spec, r containertypes.Resources) error {
	weightDevices, err := getBlkioWeightDevices(r)
	if err != nil {
		return err
	}
	readBpsDevice, err := getBlkioThrottleDevices(r.BlkioDeviceReadBps)
	if err != nil {
		return err
	}
	writeBpsDevice, err := getBlkioThrottleDevices(r.BlkioDeviceWriteBps)
	if err != nil {
		return err
	}
	readIOpsDevice, err := getBlkioThrottleDevices(r.BlkioDeviceReadIOps)
	if err != nil {
		return err
	}
	writeIOpsDevice, err := getBlkioThrottleDevices(r.BlkioDeviceWriteIOps)
	if err != nil {
		return err
	}

	memoryRes := getMemoryResources(r)
	cpuRes, err := getCPUResources(r)
	if err != nil {
		return err
	}
	blkioWeight := r.BlkioWeight

	specResources := &specs.LinuxResources{
		Memory: memoryRes,
		CPU:    cpuRes,
		BlockIO: &specs.LinuxBlockIO{
			Weight:                  &blkioWeight,
			WeightDevice:            weightDevices,
			ThrottleReadBpsDevice:   readBpsDevice,
			ThrottleWriteBpsDevice:  writeBpsDevice,
			ThrottleReadIOPSDevice:  readIOpsDevice,
			ThrottleWriteIOPSDevice: writeIOpsDevice,
		},
		Pids: &specs.LinuxPids{
			Limit: r.PidsLimit,
		},
	}

	if s.Linux.Resources != nil && len(s.Linux.Resources.Devices) > 0 {
		specResources.Devices = s.Linux.Resources.Devices
	}

	s.Linux.Resources = specResources
	return nil
}

func setDevices(s *specs.Spec, c *container.Container) error {
	// Build lists of devices allowed and created within the container.
	var devs []specs.LinuxDevice
	devPermissions := s.Linux.Resources.Devices
	if c.HostConfig.Privileged {
		hostDevices, err := devices.HostDevices()
		if err != nil {
			return err
		}
		for _, d := range hostDevices {
			devs = append(devs, oci.Device(d))
		}
		devPermissions = []specs.LinuxDeviceCgroup{
			{
				Allow:  true,
				Access: "rwm",
			},
		}
	} else {
		for _, deviceMapping := range c.HostConfig.Devices {
			d, dPermissions, err := oci.DevicesFromPath(deviceMapping.PathOnHost, deviceMapping.PathInContainer, deviceMapping.CgroupPermissions)
			if err != nil {
				return err
			}
			devs = append(devs, d...)
			devPermissions = append(devPermissions, dPermissions...)
		}

		for _, deviceCgroupRule := range c.HostConfig.DeviceCgroupRules {
			ss := deviceCgroupRuleRegex.FindAllStringSubmatch(deviceCgroupRule, -1)
			if len(ss[0]) != 5 {
				return fmt.Errorf("invalid device cgroup rule format: '%s'", deviceCgroupRule)
			}
			matches := ss[0]

			dPermissions := specs.LinuxDeviceCgroup{
				Allow:  true,
				Type:   matches[1],
				Access: matches[4],
			}
			if matches[2] == "*" {
				major := int64(-1)
				dPermissions.Major = &major
			} else {
				major, err := strconv.ParseInt(matches[2], 10, 64)
				if err != nil {
					return fmt.Errorf("invalid major value in device cgroup rule format: '%s'", deviceCgroupRule)
				}
				dPermissions.Major = &major
			}
			if matches[3] == "*" {
				minor := int64(-1)
				dPermissions.Minor = &minor
			} else {
				minor, err := strconv.ParseInt(matches[3], 10, 64)
				if err != nil {
					return fmt.Errorf("invalid minor value in device cgroup rule format: '%s'", deviceCgroupRule)
				}
				dPermissions.Minor = &minor
			}
			devPermissions = append(devPermissions, dPermissions)
		}
	}

	s.Linux.Devices = append(s.Linux.Devices, devs...)
	s.Linux.Resources.Devices = devPermissions
	return nil
}

func (daemon *Daemon) setRlimits(s *specs.Spec, c *container.Container) error {
	var rlimits []specs.POSIXRlimit

	// We want to leave the original HostConfig alone so make a copy here
	hostConfig := *c.HostConfig
	// Merge with the daemon defaults
	daemon.mergeUlimits(&hostConfig)
	for _, ul := range hostConfig.Ulimits {
		rlimits = append(rlimits, specs.POSIXRlimit{
			Type: "RLIMIT_" + strings.ToUpper(ul.Name),
			Soft: uint64(ul.Soft),
			Hard: uint64(ul.Hard),
		})
	}

	s.Process.Rlimits = rlimits
	return nil
}

func setUser(s *specs.Spec, c *container.Container) error {
	uid, gid, additionalGids, err := getUser(c, c.Config.User)
	if err != nil {
		return err
	}
	s.Process.User.UID = uid
	s.Process.User.GID = gid
	s.Process.User.AdditionalGids = additionalGids
	return nil
}

func readUserFile(c *container.Container, p string) (io.ReadCloser, error) {
	fp, err := c.GetResourcePath(p)
	if err != nil {
		return nil, err
	}
	return os.Open(fp)
}

func getUser(c *container.Container, username string) (uint32, uint32, []uint32, error) {
	passwdPath, err := user.GetPasswdPath()
	if err != nil {
		return 0, 0, nil, err
	}
	groupPath, err := user.GetGroupPath()
	if err != nil {
		return 0, 0, nil, err
	}
	passwdFile, err := readUserFile(c, passwdPath)
	if err == nil {
		defer passwdFile.Close()
	}
	groupFile, err := readUserFile(c, groupPath)
	if err == nil {
		defer groupFile.Close()
	}

	execUser, err := user.GetExecUser(username, nil, passwdFile, groupFile)
	if err != nil {
		return 0, 0, nil, err
	}

	// todo: fix this double read by a change to libcontainer/user pkg
	groupFile, err = readUserFile(c, groupPath)
	if err == nil {
		defer groupFile.Close()
	}
	var addGroups []int
	if len(c.HostConfig.GroupAdd) > 0 {
		addGroups, err = user.GetAdditionalGroups(c.HostConfig.GroupAdd, groupFile)
		if err != nil {
			return 0, 0, nil, err
		}
	}
	uid := uint32(execUser.Uid)
	gid := uint32(execUser.Gid)
	sgids := append(execUser.Sgids, addGroups...)
	var additionalGids []uint32
	for _, g := range sgids {
		additionalGids = append(additionalGids, uint32(g))
	}
	return uid, gid, additionalGids, nil
}

func setNamespace(s *specs.Spec, ns specs.LinuxNamespace) {
	for i, n := range s.Linux.Namespaces {
		if n.Type == ns.Type {
			s.Linux.Namespaces[i] = ns
			return
		}
	}
	s.Linux.Namespaces = append(s.Linux.Namespaces, ns)
}

func setCapabilities(s *specs.Spec, c *container.Container) error {
	var caplist []string
	var err error
	if c.HostConfig.Privileged {
		caplist = caps.GetAllCapabilities()
	} else {
		caplist, err = caps.TweakCapabilities(s.Process.Capabilities.Effective, c.HostConfig.CapAdd, c.HostConfig.CapDrop)
		if err != nil {
			return err
		}
	}
	s.Process.Capabilities.Effective = caplist
	s.Process.Capabilities.Bounding = caplist
	s.Process.Capabilities.Permitted = caplist
	s.Process.Capabilities.Inheritable = caplist
	return nil
}

func setNamespaces(daemon *Daemon, s *specs.Spec, c *container.Container) error {
	userNS := false
	// user
	if c.HostConfig.UsernsMode.IsPrivate() {
		uidMap := daemon.idMappings.UIDs()
		if uidMap != nil {
			userNS = true
			ns := specs.LinuxNamespace{Type: "user"}
			setNamespace(s, ns)
			s.Linux.UIDMappings = specMapping(uidMap)
			s.Linux.GIDMappings = specMapping(daemon.idMappings.GIDs())
		}
	}
	// network
	if !c.Config.NetworkDisabled {
		ns := specs.LinuxNamespace{Type: "network"}
		parts := strings.SplitN(string(c.HostConfig.NetworkMode), ":", 2)
		if parts[0] == "container" {
			nc, err := daemon.getNetworkedContainer(c.ID, c.HostConfig.NetworkMode.ConnectedContainer())
			if err != nil {
				return err
			}
			ns.Path = fmt.Sprintf("/proc/%d/ns/net", nc.State.GetPID())
			if userNS {
				// to share a net namespace, they must also share a user namespace
				nsUser := specs.LinuxNamespace{Type: "user"}
				nsUser.Path = fmt.Sprintf("/proc/%d/ns/user", nc.State.GetPID())
				setNamespace(s, nsUser)
			}
		} else if c.HostConfig.NetworkMode.IsHost() {
			ns.Path = c.NetworkSettings.SandboxKey
		}
		setNamespace(s, ns)
	}

	// ipc
	ipcMode := c.HostConfig.IpcMode
	switch {
	case ipcMode.IsContainer():
		ns := specs.LinuxNamespace{Type: "ipc"}
		ic, err := daemon.getIpcContainer(ipcMode.Container())
		if err != nil {
			return err
		}
		ns.Path = fmt.Sprintf("/proc/%d/ns/ipc", ic.State.GetPID())
		setNamespace(s, ns)
		if userNS {
			// to share an IPC namespace, they must also share a user namespace
			nsUser := specs.LinuxNamespace{Type: "user"}
			nsUser.Path = fmt.Sprintf("/proc/%d/ns/user", ic.State.GetPID())
			setNamespace(s, nsUser)
		}
	case ipcMode.IsHost():
		oci.RemoveNamespace(s, specs.LinuxNamespaceType("ipc"))
	case ipcMode.IsEmpty():
		// A container was created by an older version of the daemon.
		// The default behavior used to be what is now called "shareable".
		fallthrough
	case ipcMode.IsPrivate(), ipcMode.IsShareable(), ipcMode.IsNone():
		ns := specs.LinuxNamespace{Type: "ipc"}
		setNamespace(s, ns)
	default:
		return fmt.Errorf("Invalid IPC mode: %v", ipcMode)
	}

	// pid
	if c.HostConfig.PidMode.IsContainer() {
		ns := specs.LinuxNamespace{Type: "pid"}
		pc, err := daemon.getPidContainer(c)
		if err != nil {
			return err
		}
		ns.Path = fmt.Sprintf("/proc/%d/ns/pid", pc.State.GetPID())
		setNamespace(s, ns)
		if userNS {
			// to share a PID namespace, they must also share a user namespace
			nsUser := specs.LinuxNamespace{Type: "user"}
			nsUser.Path = fmt.Sprintf("/proc/%d/ns/user", pc.State.GetPID())
			setNamespace(s, nsUser)
		}
	} else if c.HostConfig.PidMode.IsHost() {
		oci.RemoveNamespace(s, specs.LinuxNamespaceType("pid"))
	} else {
		ns := specs.LinuxNamespace{Type: "pid"}
		setNamespace(s, ns)
	}
	// uts
	if c.HostConfig.UTSMode.IsHost() {
		oci.RemoveNamespace(s, specs.LinuxNamespaceType("uts"))
		s.Hostname = ""
	}

	return nil
}

func specMapping(s []idtools.IDMap) []specs.LinuxIDMapping {
	var ids []specs.LinuxIDMapping
	for _, item := range s {
		ids = append(ids, specs.LinuxIDMapping{
			HostID:      uint32(item.HostID),
			ContainerID: uint32(item.ContainerID),
			Size:        uint32(item.Size),
		})
	}
	return ids
}

func getMountInfo(mountinfo []*mount.Info, dir string) *mount.Info {
	for _, m := range mountinfo {
		if m.Mountpoint == dir {
			return m
		}
	}
	return nil
}

// Get the source mount point of directory passed in as argument. Also return
// optional fields.
func getSourceMount(source string) (string, string, error) {
	// Ensure any symlinks are resolved.
	sourcePath, err := filepath.EvalSymlinks(source)
	if err != nil {
		return "", "", err
	}

	mountinfos, err := mount.GetMounts()
	if err != nil {
		return "", "", err
	}

	mountinfo := getMountInfo(mountinfos, sourcePath)
	if mountinfo != nil {
		return sourcePath, mountinfo.Optional, nil
	}

	path := sourcePath
	for {
		path = filepath.Dir(path)

		mountinfo = getMountInfo(mountinfos, path)
		if mountinfo != nil {
			return path, mountinfo.Optional, nil
		}

		if path == "/" {
			break
		}
	}

	// If we are here, we did not find parent mount. Something is wrong.
	return "", "", fmt.Errorf("Could not find source mount of %s", source)
}

// Ensure mount point on which path is mounted, is shared.
func ensureShared(path string) error {
	sharedMount := false

	sourceMount, optionalOpts, err := getSourceMount(path)
	if err != nil {
		return err
	}
	// Make sure source mount point is shared.
	optsSplit := strings.Split(optionalOpts, " ")
	for _, opt := range optsSplit {
		if strings.HasPrefix(opt, "shared:") {
			sharedMount = true
			break
		}
	}

	if !sharedMount {
		return fmt.Errorf("path %s is mounted on %s but it is not a shared mount", path, sourceMount)
	}
	return nil
}

// Ensure mount point on which path is mounted, is either shared or slave.
func ensureSharedOrSlave(path string) error {
	sharedMount := false
	slaveMount := false

	sourceMount, optionalOpts, err := getSourceMount(path)
	if err != nil {
		return err
	}
	// Make sure source mount point is shared.
	optsSplit := strings.Split(optionalOpts, " ")
	for _, opt := range optsSplit {
		if strings.HasPrefix(opt, "shared:") {
			sharedMount = true
			break
		} else if strings.HasPrefix(opt, "master:") {
			slaveMount = true
			break
		}
	}

	if !sharedMount && !slaveMount {
		return fmt.Errorf("path %s is mounted on %s but it is not a shared or slave mount", path, sourceMount)
	}
	return nil
}

// Get the set of mount flags that are set on the mount that contains the given
// path and are locked by CL_UNPRIVILEGED. This is necessary to ensure that
// bind-mounting "with options" will not fail with user namespaces, due to
// kernel restrictions that require user namespace mounts to preserve
// CL_UNPRIVILEGED locked flags.
func getUnprivilegedMountFlags(path string) ([]string, error) {
	var statfs unix.Statfs_t
	if err := unix.Statfs(path, &statfs); err != nil {
		return nil, err
	}

	// The set of keys come from https://github.com/torvalds/linux/blob/v4.13/fs/namespace.c#L1034-L1048.
	unprivilegedFlags := map[uint64]string{
		unix.MS_RDONLY:     "ro",
		unix.MS_NODEV:      "nodev",
		unix.MS_NOEXEC:     "noexec",
		unix.MS_NOSUID:     "nosuid",
		unix.MS_NOATIME:    "noatime",
		unix.MS_RELATIME:   "relatime",
		unix.MS_NODIRATIME: "nodiratime",
	}

	var flags []string
	for mask, flag := range unprivilegedFlags {
		if uint64(statfs.Flags)&mask == mask {
			flags = append(flags, flag)
		}
	}

	return flags, nil
}

var (
	mountPropagationMap = map[string]int{
		"private":  mount.PRIVATE,
		"rprivate": mount.RPRIVATE,
		"shared":   mount.SHARED,
		"rshared":  mount.RSHARED,
		"slave":    mount.SLAVE,
		"rslave":   mount.RSLAVE,
	}

	mountPropagationReverseMap = map[int]string{
		mount.PRIVATE:  "private",
		mount.RPRIVATE: "rprivate",
		mount.SHARED:   "shared",
		mount.RSHARED:  "rshared",
		mount.SLAVE:    "slave",
		mount.RSLAVE:   "rslave",
	}
)

// inSlice tests whether a string is contained in a slice of strings or not.
// Comparison is case sensitive
func inSlice(slice []string, s string) bool {
	for _, ss := range slice {
		if s == ss {
			return true
		}
	}
	return false
}

func setMounts(daemon *Daemon, s *specs.Spec, c *container.Container, mounts []container.Mount) error {
	userMounts := make(map[string]struct{})
	for _, m := range mounts {
		userMounts[m.Destination] = struct{}{}
	}

	// Copy all mounts from spec to defaultMounts, except for
	//  - mounts overriden by a user supplied mount;
	//  - all mounts under /dev if a user supplied /dev is present;
	//  - /dev/shm, in case IpcMode is none.
	// While at it, also
	//  - set size for /dev/shm from shmsize.
	var defaultMounts []specs.Mount
	_, mountDev := userMounts["/dev"]
	for _, m := range s.Mounts {
		if _, ok := userMounts[m.Destination]; ok {
			// filter out mount overridden by a user supplied mount
			continue
		}
		if mountDev && strings.HasPrefix(m.Destination, "/dev/") {
			// filter out everything under /dev if /dev is user-mounted
			continue
		}

		if m.Destination == "/dev/shm" {
			if c.HostConfig.IpcMode.IsNone() {
				// filter out /dev/shm for "none" IpcMode
				continue
			}
			// set size for /dev/shm mount from spec
			sizeOpt := "size=" + strconv.FormatInt(c.HostConfig.ShmSize, 10)
			m.Options = append(m.Options, sizeOpt)
		}

		defaultMounts = append(defaultMounts, m)
	}

	s.Mounts = defaultMounts
	for _, m := range mounts {
		for _, cm := range s.Mounts {
			if cm.Destination == m.Destination {
				return duplicateMountPointError(m.Destination)
			}
		}

		if m.Source == "tmpfs" {
			data := m.Data
			parser := volume.NewParser("linux")
			options := []string{"noexec", "nosuid", "nodev", string(parser.DefaultPropagationMode())}
			if data != "" {
				options = append(options, strings.Split(data, ",")...)
			}

			merged, err := mount.MergeTmpfsOptions(options)
			if err != nil {
				return err
			}

			s.Mounts = append(s.Mounts, specs.Mount{Destination: m.Destination, Source: m.Source, Type: "tmpfs", Options: merged})
			continue
		}

		mt := specs.Mount{Destination: m.Destination, Source: m.Source, Type: "bind"}

		// Determine property of RootPropagation based on volume
		// properties. If a volume is shared, then keep root propagation
		// shared. This should work for slave and private volumes too.
		//
		// For slave volumes, it can be either [r]shared/[r]slave.
		//
		// For private volumes any root propagation value should work.
		pFlag := mountPropagationMap[m.Propagation]
		if pFlag == mount.SHARED || pFlag == mount.RSHARED {
			if err := ensureShared(m.Source); err != nil {
				return err
			}
			rootpg := mountPropagationMap[s.Linux.RootfsPropagation]
			if rootpg != mount.SHARED && rootpg != mount.RSHARED {
				s.Linux.RootfsPropagation = mountPropagationReverseMap[mount.SHARED]
			}
		} else if pFlag == mount.SLAVE || pFlag == mount.RSLAVE {
			if err := ensureSharedOrSlave(m.Source); err != nil {
				return err
			}
			rootpg := mountPropagationMap[s.Linux.RootfsPropagation]
			if rootpg != mount.SHARED && rootpg != mount.RSHARED && rootpg != mount.SLAVE && rootpg != mount.RSLAVE {
				s.Linux.RootfsPropagation = mountPropagationReverseMap[mount.RSLAVE]
			}
		}

		opts := []string{"rbind"}
		if !m.Writable {
			opts = append(opts, "ro")
		}
		if pFlag != 0 {
			opts = append(opts, mountPropagationReverseMap[pFlag])
		}

		// If we are using user namespaces, then we must make sure that we
		// don't drop any of the CL_UNPRIVILEGED "locked" flags of the source
		// "mount" when we bind-mount. The reason for this is that at the point
		// when runc sets up the root filesystem, it is already inside a user
		// namespace, and thus cannot change any flags that are locked.
		if daemon.configStore.RemappedRoot != "" {
			unprivOpts, err := getUnprivilegedMountFlags(m.Source)
			if err != nil {
				return err
			}
			opts = append(opts, unprivOpts...)
		}

		mt.Options = opts
		s.Mounts = append(s.Mounts, mt)
	}

	if s.Root.Readonly {
		for i, m := range s.Mounts {
			switch m.Destination {
			case "/proc", "/dev/pts", "/dev/mqueue", "/dev":
				continue
			}
			if _, ok := userMounts[m.Destination]; !ok {
				if !inSlice(m.Options, "ro") {
					s.Mounts[i].Options = append(s.Mounts[i].Options, "ro")
				}
			}
		}
	}

	if c.HostConfig.Privileged {
		if !s.Root.Readonly {
			// clear readonly for /sys
			for i := range s.Mounts {
				if s.Mounts[i].Destination == "/sys" {
					clearReadOnly(&s.Mounts[i])
				}
			}
		}
		s.Linux.ReadonlyPaths = nil
		s.Linux.MaskedPaths = nil
	}

	// TODO: until a kernel/mount solution exists for handling remount in a user namespace,
	// we must clear the readonly flag for the cgroups mount (@mrunalp concurs)
	if uidMap := daemon.idMappings.UIDs(); uidMap != nil || c.HostConfig.Privileged {
		for i, m := range s.Mounts {
			if m.Type == "cgroup" {
				clearReadOnly(&s.Mounts[i])
			}
		}
	}

	return nil
}

func (daemon *Daemon) populateCommonSpec(s *specs.Spec, c *container.Container) error {
	linkedEnv, err := daemon.setupLinkedContainers(c)
	if err != nil {
		return err
	}
	s.Root = &specs.Root{
		Path:     c.BaseFS.Path(),
		Readonly: c.HostConfig.ReadonlyRootfs,
	}
	if err := c.SetupWorkingDirectory(daemon.idMappings.RootPair()); err != nil {
		return err
	}
	cwd := c.Config.WorkingDir
	if len(cwd) == 0 {
		cwd = "/"
	}
	s.Process.Args = append([]string{c.Path}, c.Args...)

	// only add the custom init if it is specified and the container is running in its
	// own private pid namespace.  It does not make sense to add if it is running in the
	// host namespace or another container's pid namespace where we already have an init
	if c.HostConfig.PidMode.IsPrivate() {
		if (c.HostConfig.Init != nil && *c.HostConfig.Init) ||
			(c.HostConfig.Init == nil && daemon.configStore.Init) {
			s.Process.Args = append([]string{"/dev/init", "--", c.Path}, c.Args...)
			var path string
			if daemon.configStore.InitPath == "" {
				path, err = exec.LookPath(daemonconfig.DefaultInitBinary)
				if err != nil {
					return err
				}
			}
			if daemon.configStore.InitPath != "" {
				path = daemon.configStore.InitPath
			}
			s.Mounts = append(s.Mounts, specs.Mount{
				Destination: "/dev/init",
				Type:        "bind",
				Source:      path,
				Options:     []string{"bind", "ro"},
			})
		}
	}
	s.Process.Cwd = cwd
	s.Process.Env = c.CreateDaemonEnvironment(c.Config.Tty, linkedEnv)
	s.Process.Terminal = c.Config.Tty
	s.Hostname = c.FullHostname()

	return nil
}

func (daemon *Daemon) createSpec(c *container.Container) (*specs.Spec, error) {
	s := oci.DefaultSpec()
	if err := daemon.populateCommonSpec(&s, c); err != nil {
		return nil, err
	}

	var cgroupsPath string
	scopePrefix := "docker"
	parent := "/docker"
	useSystemd := UsingSystemd(daemon.configStore)
	if useSystemd {
		parent = "system.slice"
	}

	if c.HostConfig.CgroupParent != "" {
		parent = c.HostConfig.CgroupParent
	} else if daemon.configStore.CgroupParent != "" {
		parent = daemon.configStore.CgroupParent
	}

	if useSystemd {
		cgroupsPath = parent + ":" + scopePrefix + ":" + c.ID
		logrus.Debugf("createSpec: cgroupsPath: %s", cgroupsPath)
	} else {
		cgroupsPath = filepath.Join(parent, c.ID)
	}
	s.Linux.CgroupsPath = cgroupsPath

	if err := setResources(&s, c.HostConfig.Resources); err != nil {
		return nil, fmt.Errorf("linux runtime spec resources: %v", err)
	}
	s.Linux.Sysctl = c.HostConfig.Sysctls

	p := s.Linux.CgroupsPath
	if useSystemd {
		initPath, err := cgroups.GetInitCgroup("cpu")
		if err != nil {
			return nil, err
		}
		_, err = cgroups.GetOwnCgroup("cpu")
		if err != nil {
			return nil, err
		}
		p = filepath.Join(initPath, s.Linux.CgroupsPath)
	}

	// Clean path to guard against things like ../../../BAD
	parentPath := filepath.Dir(p)
	if !filepath.IsAbs(parentPath) {
		parentPath = filepath.Clean("/" + parentPath)
	}

	if err := daemon.initCgroupsPath(parentPath); err != nil {
		return nil, fmt.Errorf("linux init cgroups path: %v", err)
	}
	if err := setDevices(&s, c); err != nil {
		return nil, fmt.Errorf("linux runtime spec devices: %v", err)
	}
	if err := daemon.setRlimits(&s, c); err != nil {
		return nil, fmt.Errorf("linux runtime spec rlimits: %v", err)
	}
	if err := setUser(&s, c); err != nil {
		return nil, fmt.Errorf("linux spec user: %v", err)
	}
	if err := setNamespaces(daemon, &s, c); err != nil {
		return nil, fmt.Errorf("linux spec namespaces: %v", err)
	}
	if err := setCapabilities(&s, c); err != nil {
		return nil, fmt.Errorf("linux spec capabilities: %v", err)
	}
	if err := setSeccomp(daemon, &s, c); err != nil {
		return nil, fmt.Errorf("linux seccomp: %v", err)
	}

	if err := daemon.setupIpcDirs(c); err != nil {
		return nil, err
	}

	if err := daemon.setupSecretDir(c); err != nil {
		return nil, err
	}

	if err := daemon.setupConfigDir(c); err != nil {
		return nil, err
	}

	ms, err := daemon.setupMounts(c)
	if err != nil {
		return nil, err
	}

	if !c.HostConfig.IpcMode.IsPrivate() && !c.HostConfig.IpcMode.IsEmpty() {
		ms = append(ms, c.IpcMounts()...)
	}

	tmpfsMounts, err := c.TmpfsMounts()
	if err != nil {
		return nil, err
	}
	ms = append(ms, tmpfsMounts...)

	if m := c.SecretMounts(); m != nil {
		ms = append(ms, m...)
	}

	ms = append(ms, c.ConfigMounts()...)

	sort.Sort(mounts(ms))
	if err := setMounts(daemon, &s, c, ms); err != nil {
		return nil, fmt.Errorf("linux mounts: %v", err)
	}

	for _, ns := range s.Linux.Namespaces {
		if ns.Type == "network" && ns.Path == "" && !c.Config.NetworkDisabled {
			target, err := os.Readlink(filepath.Join("/proc", strconv.Itoa(os.Getpid()), "exe"))
			if err != nil {
				return nil, err
			}

			s.Hooks = &specs.Hooks{
				Prestart: []specs.Hook{{
					Path: target, // FIXME: cross-platform
					Args: []string{"libnetwork-setkey", c.ID, daemon.netController.ID()},
				}},
			}
		}
	}

	if apparmor.IsEnabled() {
		var appArmorProfile string
		if c.AppArmorProfile != "" {
			appArmorProfile = c.AppArmorProfile
		} else if c.HostConfig.Privileged {
			appArmorProfile = "unconfined"
		} else {
			appArmorProfile = "docker-default"
		}

		if appArmorProfile == "docker-default" {
			// Unattended upgrades and other fun services can unload AppArmor
			// profiles inadvertently. Since we cannot store our profile in
			// /etc/apparmor.d, nor can we practically add other ways of
			// telling the system to keep our profile loaded, in order to make
			// sure that we keep the default profile enabled we dynamically
			// reload it if necessary.
			if err := ensureDefaultAppArmorProfile(); err != nil {
				return nil, err
			}
		}

		s.Process.ApparmorProfile = appArmorProfile
	}
	s.Process.SelinuxLabel = c.GetProcessLabel()
	s.Process.NoNewPrivileges = c.NoNewPrivileges
	s.Process.OOMScoreAdj = &c.HostConfig.OomScoreAdj
	s.Linux.MountLabel = c.MountLabel

	return &s, nil
}

func clearReadOnly(m *specs.Mount) {
	var opt []string
	for _, o := range m.Options {
		if o != "ro" {
			opt = append(opt, o)
		}
	}
	m.Options = opt
}

// mergeUlimits merge the Ulimits from HostConfig with daemon defaults, and update HostConfig
func (daemon *Daemon) mergeUlimits(c *containertypes.HostConfig) {
	ulimits := c.Ulimits
	// Merge ulimits with daemon defaults
	ulIdx := make(map[string]struct{})
	for _, ul := range ulimits {
		ulIdx[ul.Name] = struct{}{}
	}
	for name, ul := range daemon.configStore.Ulimits {
		if _, exists := ulIdx[name]; !exists {
			ulimits = append(ulimits, ul)
		}
	}
	c.Ulimits = ulimits
}
