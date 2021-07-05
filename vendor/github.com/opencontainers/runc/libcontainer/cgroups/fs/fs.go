// +build linux

package fs

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
	libcontainerUtils "github.com/opencontainers/runc/libcontainer/utils"
	"github.com/pkg/errors"
	"golang.org/x/sys/unix"
)

var (
	subsystems = []subsystem{
		&CpusetGroup{},
		&DevicesGroup{},
		&MemoryGroup{},
		&CpuGroup{},
		&CpuacctGroup{},
		&PidsGroup{},
		&BlkioGroup{},
		&HugetlbGroup{},
		&NetClsGroup{},
		&NetPrioGroup{},
		&PerfEventGroup{},
		&FreezerGroup{},
		&NameGroup{GroupName: "name=systemd", Join: true},
	}
	HugePageSizes, _ = cgroups.GetHugePageSize()
)

var errSubsystemDoesNotExist = errors.New("cgroup: subsystem does not exist")

type subsystem interface {
	// Name returns the name of the subsystem.
	Name() string
	// Returns the stats, as 'stats', corresponding to the cgroup under 'path'.
	GetStats(path string, stats *cgroups.Stats) error
	// Creates and joins the cgroup represented by 'cgroupData'.
	Apply(path string, c *cgroupData) error
	// Set sets the cgroup resources.
	Set(path string, r *configs.Resources) error
}

type manager struct {
	mu       sync.Mutex
	cgroups  *configs.Cgroup
	rootless bool // ignore permission-related errors
	paths    map[string]string
}

func NewManager(cg *configs.Cgroup, paths map[string]string, rootless bool) cgroups.Manager {
	return &manager{
		cgroups:  cg,
		paths:    paths,
		rootless: rootless,
	}
}

// The absolute path to the root of the cgroup hierarchies.
var cgroupRootLock sync.Mutex
var cgroupRoot string

const defaultCgroupRoot = "/sys/fs/cgroup"

func tryDefaultCgroupRoot() string {
	var st, pst unix.Stat_t

	// (1) it should be a directory...
	err := unix.Lstat(defaultCgroupRoot, &st)
	if err != nil || st.Mode&unix.S_IFDIR == 0 {
		return ""
	}

	// (2) ... and a mount point ...
	err = unix.Lstat(filepath.Dir(defaultCgroupRoot), &pst)
	if err != nil {
		return ""
	}

	if st.Dev == pst.Dev {
		// parent dir has the same dev -- not a mount point
		return ""
	}

	// (3) ... of 'tmpfs' fs type.
	var fst unix.Statfs_t
	err = unix.Statfs(defaultCgroupRoot, &fst)
	if err != nil || fst.Type != unix.TMPFS_MAGIC {
		return ""
	}

	// (4) it should have at least 1 entry ...
	dir, err := os.Open(defaultCgroupRoot)
	if err != nil {
		return ""
	}
	names, err := dir.Readdirnames(1)
	if err != nil {
		return ""
	}
	if len(names) < 1 {
		return ""
	}
	// ... which is a cgroup mount point.
	err = unix.Statfs(filepath.Join(defaultCgroupRoot, names[0]), &fst)
	if err != nil || fst.Type != unix.CGROUP_SUPER_MAGIC {
		return ""
	}

	return defaultCgroupRoot
}

// Gets the cgroupRoot.
func getCgroupRoot() (string, error) {
	cgroupRootLock.Lock()
	defer cgroupRootLock.Unlock()

	if cgroupRoot != "" {
		return cgroupRoot, nil
	}

	// fast path
	cgroupRoot = tryDefaultCgroupRoot()
	if cgroupRoot != "" {
		return cgroupRoot, nil
	}

	// slow path: parse mountinfo
	mi, err := cgroups.GetCgroupMounts(false)
	if err != nil {
		return "", err
	}
	if len(mi) < 1 {
		return "", errors.New("no cgroup mount found in mountinfo")
	}

	// Get the first cgroup mount (e.g. "/sys/fs/cgroup/memory"),
	// use its parent directory.
	root := filepath.Dir(mi[0].Mountpoint)

	if _, err := os.Stat(root); err != nil {
		return "", err
	}

	cgroupRoot = root
	return cgroupRoot, nil
}

type cgroupData struct {
	root      string
	innerPath string
	config    *configs.Cgroup
	pid       int
}

// isIgnorableError returns whether err is a permission error (in the loose
// sense of the word). This includes EROFS (which for an unprivileged user is
// basically a permission error) and EACCES (for similar reasons) as well as
// the normal EPERM.
func isIgnorableError(rootless bool, err error) bool {
	// We do not ignore errors if we are root.
	if !rootless {
		return false
	}
	// TODO: rm errors.Cause once we switch to %w everywhere
	err = errors.Cause(err)
	// Is it an ordinary EPERM?
	if errors.Is(err, os.ErrPermission) {
		return true
	}
	// Handle some specific syscall errors.
	var errno unix.Errno
	if errors.As(err, &errno) {
		return errno == unix.EROFS || errno == unix.EPERM || errno == unix.EACCES
	}
	return false
}

func (m *manager) Apply(pid int) (err error) {
	if m.cgroups == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	c := m.cgroups
	if c.Resources.Unified != nil {
		return cgroups.ErrV1NoUnified
	}

	m.paths = make(map[string]string)
	if c.Paths != nil {
		cgMap, err := cgroups.ParseCgroupFile("/proc/self/cgroup")
		if err != nil {
			return err
		}
		for name, path := range c.Paths {
			// XXX(kolyshkin@): why this check is needed?
			if _, ok := cgMap[name]; ok {
				m.paths[name] = path
			}
		}
		return cgroups.EnterPid(m.paths, pid)
	}

	d, err := getCgroupData(m.cgroups, pid)
	if err != nil {
		return err
	}

	for _, sys := range subsystems {
		p, err := d.path(sys.Name())
		if err != nil {
			// The non-presence of the devices subsystem is
			// considered fatal for security reasons.
			if cgroups.IsNotFound(err) && (c.SkipDevices || sys.Name() != "devices") {
				continue
			}
			return err
		}
		m.paths[sys.Name()] = p

		if err := sys.Apply(p, d); err != nil {
			// In the case of rootless (including euid=0 in userns), where an
			// explicit cgroup path hasn't been set, we don't bail on error in
			// case of permission problems. Cases where limits have been set
			// (and we couldn't create our own cgroup) are handled by Set.
			if isIgnorableError(m.rootless, err) && m.cgroups.Path == "" {
				delete(m.paths, sys.Name())
				continue
			}
			return err
		}

	}
	return nil
}

func (m *manager) Destroy() error {
	if m.cgroups == nil || m.cgroups.Paths != nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return cgroups.RemovePaths(m.paths)
}

func (m *manager) Path(subsys string) string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.paths[subsys]
}

func (m *manager) GetStats() (*cgroups.Stats, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	stats := cgroups.NewStats()
	for _, sys := range subsystems {
		path := m.paths[sys.Name()]
		if path == "" {
			continue
		}
		if err := sys.GetStats(path, stats); err != nil {
			return nil, err
		}
	}
	return stats, nil
}

func (m *manager) Set(r *configs.Resources) error {
	if r == nil {
		return nil
	}

	// If Paths are set, then we are just joining cgroups paths
	// and there is no need to set any values.
	if m.cgroups != nil && m.cgroups.Paths != nil {
		return nil
	}
	if r.Unified != nil {
		return cgroups.ErrV1NoUnified
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	for _, sys := range subsystems {
		path := m.paths[sys.Name()]
		if err := sys.Set(path, r); err != nil {
			if m.rootless && sys.Name() == "devices" {
				continue
			}
			// When m.rootless is true, errors from the device subsystem are ignored because it is really not expected to work.
			// However, errors from other subsystems are not ignored.
			// see @test "runc create (rootless + limits + no cgrouppath + no permission) fails with informative error"
			if path == "" {
				// We never created a path for this cgroup, so we cannot set
				// limits for it (though we have already tried at this point).
				return fmt.Errorf("cannot set %s limit: container could not join or create cgroup", sys.Name())
			}
			return err
		}
	}

	return nil
}

// Freeze toggles the container's freezer cgroup depending on the state
// provided
func (m *manager) Freeze(state configs.FreezerState) error {
	path := m.Path("freezer")
	if m.cgroups == nil || path == "" {
		return errors.New("cannot toggle freezer: cgroups not configured for container")
	}

	prevState := m.cgroups.Resources.Freezer
	m.cgroups.Resources.Freezer = state
	freezer := &FreezerGroup{}
	if err := freezer.Set(path, m.cgroups.Resources); err != nil {
		m.cgroups.Resources.Freezer = prevState
		return err
	}
	return nil
}

func (m *manager) GetPids() ([]int, error) {
	return cgroups.GetPids(m.Path("devices"))
}

func (m *manager) GetAllPids() ([]int, error) {
	return cgroups.GetAllPids(m.Path("devices"))
}

func getCgroupData(c *configs.Cgroup, pid int) (*cgroupData, error) {
	root, err := getCgroupRoot()
	if err != nil {
		return nil, err
	}

	if (c.Name != "" || c.Parent != "") && c.Path != "" {
		return nil, errors.New("cgroup: either Path or Name and Parent should be used")
	}

	// XXX: Do not remove this code. Path safety is important! -- cyphar
	cgPath := libcontainerUtils.CleanPath(c.Path)
	cgParent := libcontainerUtils.CleanPath(c.Parent)
	cgName := libcontainerUtils.CleanPath(c.Name)

	innerPath := cgPath
	if innerPath == "" {
		innerPath = filepath.Join(cgParent, cgName)
	}

	return &cgroupData{
		root:      root,
		innerPath: innerPath,
		config:    c,
		pid:       pid,
	}, nil
}

func (raw *cgroupData) path(subsystem string) (string, error) {
	// If the cgroup name/path is absolute do not look relative to the cgroup of the init process.
	if filepath.IsAbs(raw.innerPath) {
		mnt, err := cgroups.FindCgroupMountpoint(raw.root, subsystem)
		// If we didn't mount the subsystem, there is no point we make the path.
		if err != nil {
			return "", err
		}

		// Sometimes subsystems can be mounted together as 'cpu,cpuacct'.
		return filepath.Join(raw.root, filepath.Base(mnt), raw.innerPath), nil
	}

	// Use GetOwnCgroupPath instead of GetInitCgroupPath, because the creating
	// process could in container and shared pid namespace with host, and
	// /proc/1/cgroup could point to whole other world of cgroups.
	parentPath, err := cgroups.GetOwnCgroupPath(subsystem)
	if err != nil {
		return "", err
	}

	return filepath.Join(parentPath, raw.innerPath), nil
}

func join(path string, pid int) error {
	if path == "" {
		return nil
	}
	if err := os.MkdirAll(path, 0755); err != nil {
		return err
	}
	return cgroups.WriteCgroupProc(path, pid)
}

func (m *manager) GetPaths() map[string]string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.paths
}

func (m *manager) GetCgroups() (*configs.Cgroup, error) {
	return m.cgroups, nil
}

func (m *manager) GetFreezerState() (configs.FreezerState, error) {
	dir := m.Path("freezer")
	// If the container doesn't have the freezer cgroup, say it's undefined.
	if dir == "" {
		return configs.Undefined, nil
	}
	freezer := &FreezerGroup{}
	return freezer.GetState(dir)
}

func (m *manager) Exists() bool {
	return cgroups.PathExists(m.Path("devices"))
}

func OOMKillCount(path string) (uint64, error) {
	return fscommon.GetValueByKey(path, "memory.oom_control", "oom_kill")
}

func (m *manager) OOMKillCount() (uint64, error) {
	c, err := OOMKillCount(m.Path("memory"))
	// Ignore ENOENT when rootless as it couldn't create cgroup.
	if err != nil && m.rootless && os.IsNotExist(err) {
		err = nil
	}

	return c, err
}
