package libcontainer

import (
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	securejoin "github.com/cyphar/filepath-securejoin"
	"github.com/moby/sys/mountinfo"
	"github.com/mrunalp/fileutils"
	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fs2"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runc/libcontainer/devices"
	"github.com/opencontainers/runc/libcontainer/userns"
	"github.com/opencontainers/runc/libcontainer/utils"
	"github.com/opencontainers/runtime-spec/specs-go"
	"github.com/opencontainers/selinux/go-selinux/label"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

const defaultMountFlags = unix.MS_NOEXEC | unix.MS_NOSUID | unix.MS_NODEV

type mountConfig struct {
	root            string
	label           string
	cgroup2Path     string
	rootlessCgroups bool
	cgroupns        bool
	fd              *int
}

// needsSetupDev returns true if /dev needs to be set up.
func needsSetupDev(config *configs.Config) bool {
	for _, m := range config.Mounts {
		if m.Device == "bind" && utils.CleanPath(m.Destination) == "/dev" {
			return false
		}
	}
	return true
}

// prepareRootfs sets up the devices, mount points, and filesystems for use
// inside a new mount namespace. It doesn't set anything as ro. You must call
// finalizeRootfs after this function to finish setting up the rootfs.
func prepareRootfs(pipe io.ReadWriter, iConfig *initConfig, mountFds []int) (err error) {
	config := iConfig.Config
	if err := prepareRoot(config); err != nil {
		return fmt.Errorf("error preparing rootfs: %w", err)
	}

	if mountFds != nil && len(mountFds) != len(config.Mounts) {
		return fmt.Errorf("malformed mountFds slice. Expected size: %v, got: %v. Slice: %v", len(config.Mounts), len(mountFds), mountFds)
	}

	mountConfig := &mountConfig{
		root:            config.Rootfs,
		label:           config.MountLabel,
		cgroup2Path:     iConfig.Cgroup2Path,
		rootlessCgroups: iConfig.RootlessCgroups,
		cgroupns:        config.Namespaces.Contains(configs.NEWCGROUP),
	}
	setupDev := needsSetupDev(config)
	for i, m := range config.Mounts {
		for _, precmd := range m.PremountCmds {
			if err := mountCmd(precmd); err != nil {
				return fmt.Errorf("error running premount command: %w", err)
			}
		}

		// Just before the loop we checked that if not empty, len(mountFds) == len(config.Mounts).
		// Therefore, we can access mountFds[i] without any concerns.
		if mountFds != nil && mountFds[i] != -1 {
			mountConfig.fd = &mountFds[i]
		}

		if err := mountToRootfs(m, mountConfig); err != nil {
			return fmt.Errorf("error mounting %q to rootfs at %q: %w", m.Source, m.Destination, err)
		}

		for _, postcmd := range m.PostmountCmds {
			if err := mountCmd(postcmd); err != nil {
				return fmt.Errorf("error running postmount command: %w", err)
			}
		}
	}

	if setupDev {
		if err := createDevices(config); err != nil {
			return fmt.Errorf("error creating device nodes: %w", err)
		}
		if err := setupPtmx(config); err != nil {
			return fmt.Errorf("error setting up ptmx: %w", err)
		}
		if err := setupDevSymlinks(config.Rootfs); err != nil {
			return fmt.Errorf("error setting up /dev symlinks: %w", err)
		}
	}

	// Signal the parent to run the pre-start hooks.
	// The hooks are run after the mounts are setup, but before we switch to the new
	// root, so that the old root is still available in the hooks for any mount
	// manipulations.
	// Note that iConfig.Cwd is not guaranteed to exist here.
	if err := syncParentHooks(pipe); err != nil {
		return err
	}

	// The reason these operations are done here rather than in finalizeRootfs
	// is because the console-handling code gets quite sticky if we have to set
	// up the console before doing the pivot_root(2). This is because the
	// Console API has to also work with the ExecIn case, which means that the
	// API must be able to deal with being inside as well as outside the
	// container. It's just cleaner to do this here (at the expense of the
	// operation not being perfectly split).

	if err := unix.Chdir(config.Rootfs); err != nil {
		return &os.PathError{Op: "chdir", Path: config.Rootfs, Err: err}
	}

	s := iConfig.SpecState
	s.Pid = unix.Getpid()
	s.Status = specs.StateCreating
	if err := iConfig.Config.Hooks[configs.CreateContainer].RunHooks(s); err != nil {
		return err
	}

	if config.NoPivotRoot {
		err = msMoveRoot(config.Rootfs)
	} else if config.Namespaces.Contains(configs.NEWNS) {
		err = pivotRoot(config.Rootfs)
	} else {
		err = chroot()
	}
	if err != nil {
		return fmt.Errorf("error jailing process inside rootfs: %w", err)
	}

	if setupDev {
		if err := reOpenDevNull(); err != nil {
			return fmt.Errorf("error reopening /dev/null inside container: %w", err)
		}
	}

	if cwd := iConfig.Cwd; cwd != "" {
		// Note that spec.Process.Cwd can contain unclean value like  "../../../../foo/bar...".
		// However, we are safe to call MkDirAll directly because we are in the jail here.
		if err := os.MkdirAll(cwd, 0o755); err != nil {
			return err
		}
	}

	return nil
}

// finalizeRootfs sets anything to ro if necessary. You must call
// prepareRootfs first.
func finalizeRootfs(config *configs.Config) (err error) {
	// All tmpfs mounts and /dev were previously mounted as rw
	// by mountPropagate. Remount them read-only as requested.
	for _, m := range config.Mounts {
		if m.Flags&unix.MS_RDONLY != unix.MS_RDONLY {
			continue
		}
		if m.Device == "tmpfs" || utils.CleanPath(m.Destination) == "/dev" {
			if err := remountReadonly(m); err != nil {
				return err
			}
		}
	}

	// set rootfs ( / ) as readonly
	if config.Readonlyfs {
		if err := setReadonly(); err != nil {
			return fmt.Errorf("error setting rootfs as readonly: %w", err)
		}
	}

	if config.Umask != nil {
		unix.Umask(int(*config.Umask))
	} else {
		unix.Umask(0o022)
	}
	return nil
}

// /tmp has to be mounted as private to allow MS_MOVE to work in all situations
func prepareTmp(topTmpDir string) (string, error) {
	tmpdir, err := os.MkdirTemp(topTmpDir, "runctop")
	if err != nil {
		return "", err
	}
	if err := mount(tmpdir, tmpdir, "", "bind", unix.MS_BIND, ""); err != nil {
		return "", err
	}
	if err := mount("", tmpdir, "", "", uintptr(unix.MS_PRIVATE), ""); err != nil {
		return "", err
	}
	return tmpdir, nil
}

func cleanupTmp(tmpdir string) {
	_ = unix.Unmount(tmpdir, 0)
	_ = os.RemoveAll(tmpdir)
}

func mountCmd(cmd configs.Command) error {
	command := exec.Command(cmd.Path, cmd.Args[:]...)
	command.Env = cmd.Env
	command.Dir = cmd.Dir
	if out, err := command.CombinedOutput(); err != nil {
		return fmt.Errorf("%#v failed: %s: %w", cmd, string(out), err)
	}
	return nil
}

func prepareBindMount(m *configs.Mount, rootfs string, mountFd *int) error {
	source := m.Source
	if mountFd != nil {
		source = "/proc/self/fd/" + strconv.Itoa(*mountFd)
	}

	stat, err := os.Stat(source)
	if err != nil {
		// error out if the source of a bind mount does not exist as we will be
		// unable to bind anything to it.
		return err
	}
	// ensure that the destination of the bind mount is resolved of symlinks at mount time because
	// any previous mounts can invalidate the next mount's destination.
	// this can happen when a user specifies mounts within other mounts to cause breakouts or other
	// evil stuff to try to escape the container's rootfs.
	var dest string
	if dest, err = securejoin.SecureJoin(rootfs, m.Destination); err != nil {
		return err
	}
	if err := checkProcMount(rootfs, dest, source); err != nil {
		return err
	}
	if err := createIfNotExists(dest, stat.IsDir()); err != nil {
		return err
	}

	return nil
}

func mountCgroupV1(m *configs.Mount, c *mountConfig) error {
	binds, err := getCgroupMounts(m)
	if err != nil {
		return err
	}
	var merged []string
	for _, b := range binds {
		ss := filepath.Base(b.Destination)
		if strings.Contains(ss, ",") {
			merged = append(merged, ss)
		}
	}
	tmpfs := &configs.Mount{
		Source:           "tmpfs",
		Device:           "tmpfs",
		Destination:      m.Destination,
		Flags:            defaultMountFlags,
		Data:             "mode=755",
		PropagationFlags: m.PropagationFlags,
	}

	if err := mountToRootfs(tmpfs, c); err != nil {
		return err
	}

	for _, b := range binds {
		if c.cgroupns {
			subsystemPath := filepath.Join(c.root, b.Destination)
			if err := os.MkdirAll(subsystemPath, 0o755); err != nil {
				return err
			}
			if err := utils.WithProcfd(c.root, b.Destination, func(procfd string) error {
				flags := defaultMountFlags
				if m.Flags&unix.MS_RDONLY != 0 {
					flags = flags | unix.MS_RDONLY
				}
				var (
					source = "cgroup"
					data   = filepath.Base(subsystemPath)
				)
				if data == "systemd" {
					data = cgroups.CgroupNamePrefix + data
					source = "systemd"
				}
				return mount(source, b.Destination, procfd, "cgroup", uintptr(flags), data)
			}); err != nil {
				return err
			}
		} else {
			if err := mountToRootfs(b, c); err != nil {
				return err
			}
		}
	}
	for _, mc := range merged {
		for _, ss := range strings.Split(mc, ",") {
			// symlink(2) is very dumb, it will just shove the path into
			// the link and doesn't do any checks or relative path
			// conversion. Also, don't error out if the cgroup already exists.
			if err := os.Symlink(mc, filepath.Join(c.root, m.Destination, ss)); err != nil && !os.IsExist(err) {
				return err
			}
		}
	}
	return nil
}

func mountCgroupV2(m *configs.Mount, c *mountConfig) error {
	dest, err := securejoin.SecureJoin(c.root, m.Destination)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dest, 0o755); err != nil {
		return err
	}
	return utils.WithProcfd(c.root, m.Destination, func(procfd string) error {
		if err := mount(m.Source, m.Destination, procfd, "cgroup2", uintptr(m.Flags), m.Data); err != nil {
			// when we are in UserNS but CgroupNS is not unshared, we cannot mount cgroup2 (#2158)
			if errors.Is(err, unix.EPERM) || errors.Is(err, unix.EBUSY) {
				src := fs2.UnifiedMountpoint
				if c.cgroupns && c.cgroup2Path != "" {
					// Emulate cgroupns by bind-mounting
					// the container cgroup path rather than
					// the whole /sys/fs/cgroup.
					src = c.cgroup2Path
				}
				err = mount(src, m.Destination, procfd, "", uintptr(m.Flags)|unix.MS_BIND, "")
				if c.rootlessCgroups && errors.Is(err, unix.ENOENT) {
					err = nil
				}
			}
			return err
		}
		return nil
	})
}

func doTmpfsCopyUp(m *configs.Mount, rootfs, mountLabel string) (Err error) {
	// Set up a scratch dir for the tmpfs on the host.
	tmpdir, err := prepareTmp("/tmp")
	if err != nil {
		return fmt.Errorf("tmpcopyup: failed to setup tmpdir: %w", err)
	}
	defer cleanupTmp(tmpdir)
	tmpDir, err := os.MkdirTemp(tmpdir, "runctmpdir")
	if err != nil {
		return fmt.Errorf("tmpcopyup: failed to create tmpdir: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	// Configure the *host* tmpdir as if it's the container mount. We change
	// m.Destination since we are going to mount *on the host*.
	oldDest := m.Destination
	m.Destination = tmpDir
	err = mountPropagate(m, "/", mountLabel, nil)
	m.Destination = oldDest
	if err != nil {
		return err
	}
	defer func() {
		if Err != nil {
			if err := unmount(tmpDir, unix.MNT_DETACH); err != nil {
				logrus.Warnf("tmpcopyup: %v", err)
			}
		}
	}()

	return utils.WithProcfd(rootfs, m.Destination, func(procfd string) (Err error) {
		// Copy the container data to the host tmpdir. We append "/" to force
		// CopyDirectory to resolve the symlink rather than trying to copy the
		// symlink itself.
		if err := fileutils.CopyDirectory(procfd+"/", tmpDir); err != nil {
			return fmt.Errorf("tmpcopyup: failed to copy %s to %s (%s): %w", m.Destination, procfd, tmpDir, err)
		}
		// Now move the mount into the container.
		if err := mount(tmpDir, m.Destination, procfd, "", unix.MS_MOVE, ""); err != nil {
			return fmt.Errorf("tmpcopyup: failed to move mount: %w", err)
		}
		return nil
	})
}

func mountToRootfs(m *configs.Mount, c *mountConfig) error {
	rootfs := c.root
	mountLabel := c.label
	mountFd := c.fd
	dest, err := securejoin.SecureJoin(rootfs, m.Destination)
	if err != nil {
		return err
	}

	switch m.Device {
	case "proc", "sysfs":
		// If the destination already exists and is not a directory, we bail
		// out This is to avoid mounting through a symlink or similar -- which
		// has been a "fun" attack scenario in the past.
		// TODO: This won't be necessary once we switch to libpathrs and we can
		//       stop all of these symlink-exchange attacks.
		if fi, err := os.Lstat(dest); err != nil {
			if !os.IsNotExist(err) {
				return err
			}
		} else if fi.Mode()&os.ModeDir == 0 {
			return fmt.Errorf("filesystem %q must be mounted on ordinary directory", m.Device)
		}
		if err := os.MkdirAll(dest, 0o755); err != nil {
			return err
		}
		// Selinux kernels do not support labeling of /proc or /sys
		return mountPropagate(m, rootfs, "", nil)
	case "mqueue":
		if err := os.MkdirAll(dest, 0o755); err != nil {
			return err
		}
		if err := mountPropagate(m, rootfs, "", nil); err != nil {
			return err
		}
		return label.SetFileLabel(dest, mountLabel)
	case "tmpfs":
		stat, err := os.Stat(dest)
		if err != nil {
			if err := os.MkdirAll(dest, 0o755); err != nil {
				return err
			}
		}

		if m.Extensions&configs.EXT_COPYUP == configs.EXT_COPYUP {
			err = doTmpfsCopyUp(m, rootfs, mountLabel)
		} else {
			err = mountPropagate(m, rootfs, mountLabel, nil)
		}

		if err != nil {
			return err
		}

		if stat != nil {
			if err = os.Chmod(dest, stat.Mode()); err != nil {
				return err
			}
		}
		return nil
	case "bind":
		if err := prepareBindMount(m, rootfs, mountFd); err != nil {
			return err
		}
		if err := mountPropagate(m, rootfs, mountLabel, mountFd); err != nil {
			return err
		}
		// bind mount won't change mount options, we need remount to make mount options effective.
		// first check that we have non-default options required before attempting a remount
		if m.Flags&^(unix.MS_REC|unix.MS_REMOUNT|unix.MS_BIND) != 0 {
			// only remount if unique mount options are set
			if err := remount(m, rootfs, mountFd); err != nil {
				return err
			}
		}

		if m.Relabel != "" {
			if err := label.Validate(m.Relabel); err != nil {
				return err
			}
			shared := label.IsShared(m.Relabel)
			if err := label.Relabel(m.Source, mountLabel, shared); err != nil {
				return err
			}
		}
	case "cgroup":
		if cgroups.IsCgroup2UnifiedMode() {
			return mountCgroupV2(m, c)
		}
		return mountCgroupV1(m, c)
	default:
		if err := checkProcMount(rootfs, dest, m.Source); err != nil {
			return err
		}
		if err := os.MkdirAll(dest, 0o755); err != nil {
			return err
		}
		return mountPropagate(m, rootfs, mountLabel, mountFd)
	}
	if err := setRecAttr(m, rootfs); err != nil {
		return err
	}
	return nil
}

func getCgroupMounts(m *configs.Mount) ([]*configs.Mount, error) {
	mounts, err := cgroups.GetCgroupMounts(false)
	if err != nil {
		return nil, err
	}

	cgroupPaths, err := cgroups.ParseCgroupFile("/proc/self/cgroup")
	if err != nil {
		return nil, err
	}

	var binds []*configs.Mount

	for _, mm := range mounts {
		dir, err := mm.GetOwnCgroup(cgroupPaths)
		if err != nil {
			return nil, err
		}
		relDir, err := filepath.Rel(mm.Root, dir)
		if err != nil {
			return nil, err
		}
		binds = append(binds, &configs.Mount{
			Device:           "bind",
			Source:           filepath.Join(mm.Mountpoint, relDir),
			Destination:      filepath.Join(m.Destination, filepath.Base(mm.Mountpoint)),
			Flags:            unix.MS_BIND | unix.MS_REC | m.Flags,
			PropagationFlags: m.PropagationFlags,
		})
	}

	return binds, nil
}

// checkProcMount checks to ensure that the mount destination is not over the top of /proc.
// dest is required to be an abs path and have any symlinks resolved before calling this function.
//
// if source is nil, don't stat the filesystem.  This is used for restore of a checkpoint.
func checkProcMount(rootfs, dest, source string) error {
	const procPath = "/proc"
	path, err := filepath.Rel(filepath.Join(rootfs, procPath), dest)
	if err != nil {
		return err
	}
	// pass if the mount path is located outside of /proc
	if strings.HasPrefix(path, "..") {
		return nil
	}
	if path == "." {
		// an empty source is pasted on restore
		if source == "" {
			return nil
		}
		// only allow a mount on-top of proc if it's source is "proc"
		isproc, err := isProc(source)
		if err != nil {
			return err
		}
		// pass if the mount is happening on top of /proc and the source of
		// the mount is a proc filesystem
		if isproc {
			return nil
		}
		return fmt.Errorf("%q cannot be mounted because it is not of type proc", dest)
	}

	// Here dest is definitely under /proc. Do not allow those,
	// except for a few specific entries emulated by lxcfs.
	validProcMounts := []string{
		"/proc/cpuinfo",
		"/proc/diskstats",
		"/proc/meminfo",
		"/proc/stat",
		"/proc/swaps",
		"/proc/uptime",
		"/proc/loadavg",
		"/proc/slabinfo",
		"/proc/net/dev",
	}
	for _, valid := range validProcMounts {
		path, err := filepath.Rel(filepath.Join(rootfs, valid), dest)
		if err != nil {
			return err
		}
		if path == "." {
			return nil
		}
	}

	return fmt.Errorf("%q cannot be mounted because it is inside /proc", dest)
}

func isProc(path string) (bool, error) {
	var s unix.Statfs_t
	if err := unix.Statfs(path, &s); err != nil {
		return false, &os.PathError{Op: "statfs", Path: path, Err: err}
	}
	return s.Type == unix.PROC_SUPER_MAGIC, nil
}

func setupDevSymlinks(rootfs string) error {
	links := [][2]string{
		{"/proc/self/fd", "/dev/fd"},
		{"/proc/self/fd/0", "/dev/stdin"},
		{"/proc/self/fd/1", "/dev/stdout"},
		{"/proc/self/fd/2", "/dev/stderr"},
	}
	// kcore support can be toggled with CONFIG_PROC_KCORE; only create a symlink
	// in /dev if it exists in /proc.
	if _, err := os.Stat("/proc/kcore"); err == nil {
		links = append(links, [2]string{"/proc/kcore", "/dev/core"})
	}
	for _, link := range links {
		var (
			src = link[0]
			dst = filepath.Join(rootfs, link[1])
		)
		if err := os.Symlink(src, dst); err != nil && !os.IsExist(err) {
			return err
		}
	}
	return nil
}

// If stdin, stdout, and/or stderr are pointing to `/dev/null` in the parent's rootfs
// this method will make them point to `/dev/null` in this container's rootfs.  This
// needs to be called after we chroot/pivot into the container's rootfs so that any
// symlinks are resolved locally.
func reOpenDevNull() error {
	var stat, devNullStat unix.Stat_t
	file, err := os.OpenFile("/dev/null", os.O_RDWR, 0)
	if err != nil {
		return err
	}
	defer file.Close() //nolint: errcheck
	if err := unix.Fstat(int(file.Fd()), &devNullStat); err != nil {
		return &os.PathError{Op: "fstat", Path: file.Name(), Err: err}
	}
	for fd := 0; fd < 3; fd++ {
		if err := unix.Fstat(fd, &stat); err != nil {
			return &os.PathError{Op: "fstat", Path: "fd " + strconv.Itoa(fd), Err: err}
		}
		if stat.Rdev == devNullStat.Rdev {
			// Close and re-open the fd.
			if err := unix.Dup3(int(file.Fd()), fd, 0); err != nil {
				return &os.PathError{
					Op:   "dup3",
					Path: "fd " + strconv.Itoa(int(file.Fd())),
					Err:  err,
				}
			}
		}
	}
	return nil
}

// Create the device nodes in the container.
func createDevices(config *configs.Config) error {
	useBindMount := userns.RunningInUserNS() || config.Namespaces.Contains(configs.NEWUSER)
	oldMask := unix.Umask(0o000)
	for _, node := range config.Devices {

		// The /dev/ptmx device is setup by setupPtmx()
		if utils.CleanPath(node.Path) == "/dev/ptmx" {
			continue
		}

		// containers running in a user namespace are not allowed to mknod
		// devices so we can just bind mount it from the host.
		if err := createDeviceNode(config.Rootfs, node, useBindMount); err != nil {
			unix.Umask(oldMask)
			return err
		}
	}
	unix.Umask(oldMask)
	return nil
}

func bindMountDeviceNode(rootfs, dest string, node *devices.Device) error {
	f, err := os.Create(dest)
	if err != nil && !os.IsExist(err) {
		return err
	}
	if f != nil {
		_ = f.Close()
	}
	return utils.WithProcfd(rootfs, dest, func(procfd string) error {
		return mount(node.Path, dest, procfd, "bind", unix.MS_BIND, "")
	})
}

// Creates the device node in the rootfs of the container.
func createDeviceNode(rootfs string, node *devices.Device, bind bool) error {
	if node.Path == "" {
		// The node only exists for cgroup reasons, ignore it here.
		return nil
	}
	dest, err := securejoin.SecureJoin(rootfs, node.Path)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}
	if bind {
		return bindMountDeviceNode(rootfs, dest, node)
	}
	if err := mknodDevice(dest, node); err != nil {
		if errors.Is(err, os.ErrExist) {
			return nil
		} else if errors.Is(err, os.ErrPermission) {
			return bindMountDeviceNode(rootfs, dest, node)
		}
		return err
	}
	return nil
}

func mknodDevice(dest string, node *devices.Device) error {
	fileMode := node.FileMode
	switch node.Type {
	case devices.BlockDevice:
		fileMode |= unix.S_IFBLK
	case devices.CharDevice:
		fileMode |= unix.S_IFCHR
	case devices.FifoDevice:
		fileMode |= unix.S_IFIFO
	default:
		return fmt.Errorf("%c is not a valid device type for device %s", node.Type, node.Path)
	}
	dev, err := node.Mkdev()
	if err != nil {
		return err
	}
	if err := unix.Mknod(dest, uint32(fileMode), int(dev)); err != nil {
		return &os.PathError{Op: "mknod", Path: dest, Err: err}
	}
	return os.Chown(dest, int(node.Uid), int(node.Gid))
}

// Get the parent mount point of directory passed in as argument. Also return
// optional fields.
func getParentMount(rootfs string) (string, string, error) {
	mi, err := mountinfo.GetMounts(mountinfo.ParentsFilter(rootfs))
	if err != nil {
		return "", "", err
	}
	if len(mi) < 1 {
		return "", "", fmt.Errorf("could not find parent mount of %s", rootfs)
	}

	// find the longest mount point
	var idx, maxlen int
	for i := range mi {
		if len(mi[i].Mountpoint) > maxlen {
			maxlen = len(mi[i].Mountpoint)
			idx = i
		}
	}
	return mi[idx].Mountpoint, mi[idx].Optional, nil
}

// Make parent mount private if it was shared
func rootfsParentMountPrivate(rootfs string) error {
	sharedMount := false

	parentMount, optionalOpts, err := getParentMount(rootfs)
	if err != nil {
		return err
	}

	optsSplit := strings.Split(optionalOpts, " ")
	for _, opt := range optsSplit {
		if strings.HasPrefix(opt, "shared:") {
			sharedMount = true
			break
		}
	}

	// Make parent mount PRIVATE if it was shared. It is needed for two
	// reasons. First of all pivot_root() will fail if parent mount is
	// shared. Secondly when we bind mount rootfs it will propagate to
	// parent namespace and we don't want that to happen.
	if sharedMount {
		return mount("", parentMount, "", "", unix.MS_PRIVATE, "")
	}

	return nil
}

func prepareRoot(config *configs.Config) error {
	flag := unix.MS_SLAVE | unix.MS_REC
	if config.RootPropagation != 0 {
		flag = config.RootPropagation
	}
	if err := mount("", "/", "", "", uintptr(flag), ""); err != nil {
		return err
	}

	// Make parent mount private to make sure following bind mount does
	// not propagate in other namespaces. Also it will help with kernel
	// check pass in pivot_root. (IS_SHARED(new_mnt->mnt_parent))
	if err := rootfsParentMountPrivate(config.Rootfs); err != nil {
		return err
	}

	return mount(config.Rootfs, config.Rootfs, "", "bind", unix.MS_BIND|unix.MS_REC, "")
}

func setReadonly() error {
	flags := uintptr(unix.MS_BIND | unix.MS_REMOUNT | unix.MS_RDONLY)

	err := mount("", "/", "", "", flags, "")
	if err == nil {
		return nil
	}
	var s unix.Statfs_t
	if err := unix.Statfs("/", &s); err != nil {
		return &os.PathError{Op: "statfs", Path: "/", Err: err}
	}
	flags |= uintptr(s.Flags)
	return mount("", "/", "", "", flags, "")
}

func setupPtmx(config *configs.Config) error {
	ptmx := filepath.Join(config.Rootfs, "dev/ptmx")
	if err := os.Remove(ptmx); err != nil && !os.IsNotExist(err) {
		return err
	}
	if err := os.Symlink("pts/ptmx", ptmx); err != nil {
		return err
	}
	return nil
}

// pivotRoot will call pivot_root such that rootfs becomes the new root
// filesystem, and everything else is cleaned up.
func pivotRoot(rootfs string) error {
	// While the documentation may claim otherwise, pivot_root(".", ".") is
	// actually valid. What this results in is / being the new root but
	// /proc/self/cwd being the old root. Since we can play around with the cwd
	// with pivot_root this allows us to pivot without creating directories in
	// the rootfs. Shout-outs to the LXC developers for giving us this idea.

	oldroot, err := unix.Open("/", unix.O_DIRECTORY|unix.O_RDONLY, 0)
	if err != nil {
		return &os.PathError{Op: "open", Path: "/", Err: err}
	}
	defer unix.Close(oldroot) //nolint: errcheck

	newroot, err := unix.Open(rootfs, unix.O_DIRECTORY|unix.O_RDONLY, 0)
	if err != nil {
		return &os.PathError{Op: "open", Path: rootfs, Err: err}
	}
	defer unix.Close(newroot) //nolint: errcheck

	// Change to the new root so that the pivot_root actually acts on it.
	if err := unix.Fchdir(newroot); err != nil {
		return &os.PathError{Op: "fchdir", Path: "fd " + strconv.Itoa(newroot), Err: err}
	}

	if err := unix.PivotRoot(".", "."); err != nil {
		return &os.PathError{Op: "pivot_root", Path: ".", Err: err}
	}

	// Currently our "." is oldroot (according to the current kernel code).
	// However, purely for safety, we will fchdir(oldroot) since there isn't
	// really any guarantee from the kernel what /proc/self/cwd will be after a
	// pivot_root(2).

	if err := unix.Fchdir(oldroot); err != nil {
		return &os.PathError{Op: "fchdir", Path: "fd " + strconv.Itoa(oldroot), Err: err}
	}

	// Make oldroot rslave to make sure our unmounts don't propagate to the
	// host (and thus bork the machine). We don't use rprivate because this is
	// known to cause issues due to races where we still have a reference to a
	// mount while a process in the host namespace are trying to operate on
	// something they think has no mounts (devicemapper in particular).
	if err := mount("", ".", "", "", unix.MS_SLAVE|unix.MS_REC, ""); err != nil {
		return err
	}
	// Perform the unmount. MNT_DETACH allows us to unmount /proc/self/cwd.
	if err := unmount(".", unix.MNT_DETACH); err != nil {
		return err
	}

	// Switch back to our shiny new root.
	if err := unix.Chdir("/"); err != nil {
		return &os.PathError{Op: "chdir", Path: "/", Err: err}
	}
	return nil
}

func msMoveRoot(rootfs string) error {
	// Before we move the root and chroot we have to mask all "full" sysfs and
	// procfs mounts which exist on the host. This is because while the kernel
	// has protections against mounting procfs if it has masks, when using
	// chroot(2) the *host* procfs mount is still reachable in the mount
	// namespace and the kernel permits procfs mounts inside --no-pivot
	// containers.
	//
	// Users shouldn't be using --no-pivot except in exceptional circumstances,
	// but to avoid such a trivial security flaw we apply a best-effort
	// protection here. The kernel only allows a mount of a pseudo-filesystem
	// like procfs or sysfs if there is a *full* mount (the root of the
	// filesystem is mounted) without any other locked mount points covering a
	// subtree of the mount.
	//
	// So we try to unmount (or mount tmpfs on top of) any mountpoint which is
	// a full mount of either sysfs or procfs (since those are the most
	// concerning filesystems to us).
	mountinfos, err := mountinfo.GetMounts(func(info *mountinfo.Info) (skip, stop bool) {
		// Collect every sysfs and procfs filesystem, except for those which
		// are non-full mounts or are inside the rootfs of the container.
		if info.Root != "/" ||
			(info.FSType != "proc" && info.FSType != "sysfs") ||
			strings.HasPrefix(info.Mountpoint, rootfs) {
			skip = true
		}
		return
	})
	if err != nil {
		return err
	}
	for _, info := range mountinfos {
		p := info.Mountpoint
		// Be sure umount events are not propagated to the host.
		if err := mount("", p, "", "", unix.MS_SLAVE|unix.MS_REC, ""); err != nil {
			if errors.Is(err, unix.ENOENT) {
				// If the mountpoint doesn't exist that means that we've
				// already blasted away some parent directory of the mountpoint
				// and so we don't care about this error.
				continue
			}
			return err
		}
		if err := unmount(p, unix.MNT_DETACH); err != nil {
			if !errors.Is(err, unix.EINVAL) && !errors.Is(err, unix.EPERM) {
				return err
			} else {
				// If we have not privileges for umounting (e.g. rootless), then
				// cover the path.
				if err := mount("tmpfs", p, "", "tmpfs", 0, ""); err != nil {
					return err
				}
			}
		}
	}

	// Move the rootfs on top of "/" in our mount namespace.
	if err := mount(rootfs, "/", "", "", unix.MS_MOVE, ""); err != nil {
		return err
	}
	return chroot()
}

func chroot() error {
	if err := unix.Chroot("."); err != nil {
		return &os.PathError{Op: "chroot", Path: ".", Err: err}
	}
	if err := unix.Chdir("/"); err != nil {
		return &os.PathError{Op: "chdir", Path: "/", Err: err}
	}
	return nil
}

// createIfNotExists creates a file or a directory only if it does not already exist.
func createIfNotExists(path string, isDir bool) error {
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			if isDir {
				return os.MkdirAll(path, 0o755)
			}
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				return err
			}
			f, err := os.OpenFile(path, os.O_CREATE, 0o755)
			if err != nil {
				return err
			}
			_ = f.Close()
		}
	}
	return nil
}

// readonlyPath will make a path read only.
func readonlyPath(path string) error {
	if err := mount(path, path, "", "", unix.MS_BIND|unix.MS_REC, ""); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}

	var s unix.Statfs_t
	if err := unix.Statfs(path, &s); err != nil {
		return &os.PathError{Op: "statfs", Path: path, Err: err}
	}
	flags := uintptr(s.Flags) & (unix.MS_NOSUID | unix.MS_NODEV | unix.MS_NOEXEC)

	if err := mount(path, path, "", "", flags|unix.MS_BIND|unix.MS_REMOUNT|unix.MS_RDONLY, ""); err != nil {
		return err
	}

	return nil
}

// remountReadonly will remount an existing mount point and ensure that it is read-only.
func remountReadonly(m *configs.Mount) error {
	var (
		dest  = m.Destination
		flags = m.Flags
	)
	for i := 0; i < 5; i++ {
		// There is a special case in the kernel for
		// MS_REMOUNT | MS_BIND, which allows us to change only the
		// flags even as an unprivileged user (i.e. user namespace)
		// assuming we don't drop any security related flags (nodev,
		// nosuid, etc.). So, let's use that case so that we can do
		// this re-mount without failing in a userns.
		flags |= unix.MS_REMOUNT | unix.MS_BIND | unix.MS_RDONLY
		if err := mount("", dest, "", "", uintptr(flags), ""); err != nil {
			if errors.Is(err, unix.EBUSY) {
				time.Sleep(100 * time.Millisecond)
				continue
			}
			return err
		}
		return nil
	}
	return fmt.Errorf("unable to mount %s as readonly max retries reached", dest)
}

// maskPath masks the top of the specified path inside a container to avoid
// security issues from processes reading information from non-namespace aware
// mounts ( proc/kcore ).
// For files, maskPath bind mounts /dev/null over the top of the specified path.
// For directories, maskPath mounts read-only tmpfs over the top of the specified path.
func maskPath(path string, mountLabel string) error {
	if err := mount("/dev/null", path, "", "", unix.MS_BIND, ""); err != nil && !errors.Is(err, os.ErrNotExist) {
		if errors.Is(err, unix.ENOTDIR) {
			return mount("tmpfs", path, "", "tmpfs", unix.MS_RDONLY, label.FormatMountLabel("", mountLabel))
		}
		return err
	}
	return nil
}

// writeSystemProperty writes the value to a path under /proc/sys as determined from the key.
// For e.g. net.ipv4.ip_forward translated to /proc/sys/net/ipv4/ip_forward.
func writeSystemProperty(key, value string) error {
	keyPath := strings.Replace(key, ".", "/", -1)
	return os.WriteFile(path.Join("/proc/sys", keyPath), []byte(value), 0o644)
}

func remount(m *configs.Mount, rootfs string, mountFd *int) error {
	source := m.Source
	if mountFd != nil {
		source = "/proc/self/fd/" + strconv.Itoa(*mountFd)
	}

	return utils.WithProcfd(rootfs, m.Destination, func(procfd string) error {
		flags := uintptr(m.Flags | unix.MS_REMOUNT)
		err := mount(source, m.Destination, procfd, m.Device, flags, "")
		if err == nil {
			return nil
		}
		// Check if the source has ro flag...
		var s unix.Statfs_t
		if err := unix.Statfs(source, &s); err != nil {
			return &os.PathError{Op: "statfs", Path: source, Err: err}
		}
		if s.Flags&unix.MS_RDONLY != unix.MS_RDONLY {
			return err
		}
		// ... and retry the mount with ro flag set.
		flags |= unix.MS_RDONLY
		return mount(source, m.Destination, procfd, m.Device, flags, "")
	})
}

// Do the mount operation followed by additional mounts required to take care
// of propagation flags. This will always be scoped inside the container rootfs.
func mountPropagate(m *configs.Mount, rootfs string, mountLabel string, mountFd *int) error {
	var (
		data  = label.FormatMountLabel(m.Data, mountLabel)
		flags = m.Flags
	)
	// Delay mounting the filesystem read-only if we need to do further
	// operations on it. We need to set up files in "/dev", and other tmpfs
	// mounts may need to be chmod-ed after mounting. These mounts will be
	// remounted ro later in finalizeRootfs(), if necessary.
	if m.Device == "tmpfs" || utils.CleanPath(m.Destination) == "/dev" {
		flags &= ^unix.MS_RDONLY
	}

	// Because the destination is inside a container path which might be
	// mutating underneath us, we verify that we are actually going to mount
	// inside the container with WithProcfd() -- mounting through a procfd
	// mounts on the target.
	source := m.Source
	if mountFd != nil {
		source = "/proc/self/fd/" + strconv.Itoa(*mountFd)
	}

	if err := utils.WithProcfd(rootfs, m.Destination, func(procfd string) error {
		return mount(source, m.Destination, procfd, m.Device, uintptr(flags), data)
	}); err != nil {
		return err
	}
	// We have to apply mount propagation flags in a separate WithProcfd() call
	// because the previous call invalidates the passed procfd -- the mount
	// target needs to be re-opened.
	if err := utils.WithProcfd(rootfs, m.Destination, func(procfd string) error {
		for _, pflag := range m.PropagationFlags {
			if err := mount("", m.Destination, procfd, "", uintptr(pflag), ""); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		return fmt.Errorf("change mount propagation through procfd: %w", err)
	}
	return nil
}

func setRecAttr(m *configs.Mount, rootfs string) error {
	if m.RecAttr == nil {
		return nil
	}
	return utils.WithProcfd(rootfs, m.Destination, func(procfd string) error {
		return unix.MountSetattr(-1, procfd, unix.AT_RECURSIVE, m.RecAttr)
	})
}
