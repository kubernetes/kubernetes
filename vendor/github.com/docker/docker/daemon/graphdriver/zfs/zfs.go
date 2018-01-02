// +build linux freebsd solaris

package zfs

import (
	"fmt"
	"os"
	"os/exec"
	"path"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/docker/docker/daemon/graphdriver"
	"github.com/docker/docker/pkg/idtools"
	"github.com/docker/docker/pkg/mount"
	"github.com/docker/docker/pkg/parsers"
	zfs "github.com/mistifyio/go-zfs"
	"github.com/opencontainers/selinux/go-selinux/label"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

type zfsOptions struct {
	fsName    string
	mountPath string
}

func init() {
	graphdriver.Register("zfs", Init)
}

// Logger returns a zfs logger implementation.
type Logger struct{}

// Log wraps log message from ZFS driver with a prefix '[zfs]'.
func (*Logger) Log(cmd []string) {
	logrus.Debugf("[zfs] %s", strings.Join(cmd, " "))
}

// Init returns a new ZFS driver.
// It takes base mount path and an array of options which are represented as key value pairs.
// Each option is in the for key=value. 'zfs.fsname' is expected to be a valid key in the options.
func Init(base string, opt []string, uidMaps, gidMaps []idtools.IDMap) (graphdriver.Driver, error) {
	var err error

	if _, err := exec.LookPath("zfs"); err != nil {
		logrus.Debugf("[zfs] zfs command is not available: %v", err)
		return nil, graphdriver.ErrPrerequisites
	}

	file, err := os.OpenFile("/dev/zfs", os.O_RDWR, 600)
	if err != nil {
		logrus.Debugf("[zfs] cannot open /dev/zfs: %v", err)
		return nil, graphdriver.ErrPrerequisites
	}
	defer file.Close()

	options, err := parseOptions(opt)
	if err != nil {
		return nil, err
	}
	options.mountPath = base

	rootdir := path.Dir(base)

	if options.fsName == "" {
		err = checkRootdirFs(rootdir)
		if err != nil {
			return nil, err
		}
	}

	if options.fsName == "" {
		options.fsName, err = lookupZfsDataset(rootdir)
		if err != nil {
			return nil, err
		}
	}

	zfs.SetLogger(new(Logger))

	filesystems, err := zfs.Filesystems(options.fsName)
	if err != nil {
		return nil, fmt.Errorf("Cannot find root filesystem %s: %v", options.fsName, err)
	}

	filesystemsCache := make(map[string]bool, len(filesystems))
	var rootDataset *zfs.Dataset
	for _, fs := range filesystems {
		if fs.Name == options.fsName {
			rootDataset = fs
		}
		filesystemsCache[fs.Name] = true
	}

	if rootDataset == nil {
		return nil, fmt.Errorf("BUG: zfs get all -t filesystem -rHp '%s' should contain '%s'", options.fsName, options.fsName)
	}

	rootUID, rootGID, err := idtools.GetRootUIDGID(uidMaps, gidMaps)
	if err != nil {
		return nil, fmt.Errorf("Failed to get root uid/guid: %v", err)
	}
	if err := idtools.MkdirAllAs(base, 0700, rootUID, rootGID); err != nil {
		return nil, fmt.Errorf("Failed to create '%s': %v", base, err)
	}

	if err := mount.MakePrivate(base); err != nil {
		return nil, err
	}
	d := &Driver{
		dataset:          rootDataset,
		options:          options,
		filesystemsCache: filesystemsCache,
		uidMaps:          uidMaps,
		gidMaps:          gidMaps,
		ctr:              graphdriver.NewRefCounter(graphdriver.NewDefaultChecker()),
	}
	return graphdriver.NewNaiveDiffDriver(d, uidMaps, gidMaps), nil
}

func parseOptions(opt []string) (zfsOptions, error) {
	var options zfsOptions
	options.fsName = ""
	for _, option := range opt {
		key, val, err := parsers.ParseKeyValueOpt(option)
		if err != nil {
			return options, err
		}
		key = strings.ToLower(key)
		switch key {
		case "zfs.fsname":
			options.fsName = val
		default:
			return options, fmt.Errorf("Unknown option %s", key)
		}
	}
	return options, nil
}

func lookupZfsDataset(rootdir string) (string, error) {
	var stat unix.Stat_t
	if err := unix.Stat(rootdir, &stat); err != nil {
		return "", fmt.Errorf("Failed to access '%s': %s", rootdir, err)
	}
	wantedDev := stat.Dev

	mounts, err := mount.GetMounts()
	if err != nil {
		return "", err
	}
	for _, m := range mounts {
		if err := unix.Stat(m.Mountpoint, &stat); err != nil {
			logrus.Debugf("[zfs] failed to stat '%s' while scanning for zfs mount: %v", m.Mountpoint, err)
			continue // may fail on fuse file systems
		}

		if stat.Dev == wantedDev && m.Fstype == "zfs" {
			return m.Source, nil
		}
	}

	return "", fmt.Errorf("Failed to find zfs dataset mounted on '%s' in /proc/mounts", rootdir)
}

// Driver holds information about the driver, such as zfs dataset, options and cache.
type Driver struct {
	dataset          *zfs.Dataset
	options          zfsOptions
	sync.Mutex       // protects filesystem cache against concurrent access
	filesystemsCache map[string]bool
	uidMaps          []idtools.IDMap
	gidMaps          []idtools.IDMap
	ctr              *graphdriver.RefCounter
}

func (d *Driver) String() string {
	return "zfs"
}

// Cleanup is used to implement graphdriver.ProtoDriver. There is no cleanup required for this driver.
func (d *Driver) Cleanup() error {
	return nil
}

// Status returns information about the ZFS filesystem. It returns a two dimensional array of information
// such as pool name, dataset name, disk usage, parent quota and compression used.
// Currently it return 'Zpool', 'Zpool Health', 'Parent Dataset', 'Space Used By Parent',
// 'Space Available', 'Parent Quota' and 'Compression'.
func (d *Driver) Status() [][2]string {
	parts := strings.Split(d.dataset.Name, "/")
	pool, err := zfs.GetZpool(parts[0])

	var poolName, poolHealth string
	if err == nil {
		poolName = pool.Name
		poolHealth = pool.Health
	} else {
		poolName = fmt.Sprintf("error while getting pool information %v", err)
		poolHealth = "not available"
	}

	quota := "no"
	if d.dataset.Quota != 0 {
		quota = strconv.FormatUint(d.dataset.Quota, 10)
	}

	return [][2]string{
		{"Zpool", poolName},
		{"Zpool Health", poolHealth},
		{"Parent Dataset", d.dataset.Name},
		{"Space Used By Parent", strconv.FormatUint(d.dataset.Used, 10)},
		{"Space Available", strconv.FormatUint(d.dataset.Avail, 10)},
		{"Parent Quota", quota},
		{"Compression", d.dataset.Compression},
	}
}

// GetMetadata returns image/container metadata related to graph driver
func (d *Driver) GetMetadata(id string) (map[string]string, error) {
	return map[string]string{
		"Mountpoint": d.mountPath(id),
		"Dataset":    d.zfsPath(id),
	}, nil
}

func (d *Driver) cloneFilesystem(name, parentName string) error {
	snapshotName := fmt.Sprintf("%d", time.Now().Nanosecond())
	parentDataset := zfs.Dataset{Name: parentName}
	snapshot, err := parentDataset.Snapshot(snapshotName /*recursive */, false)
	if err != nil {
		return err
	}

	_, err = snapshot.Clone(name, map[string]string{"mountpoint": "legacy"})
	if err == nil {
		d.Lock()
		d.filesystemsCache[name] = true
		d.Unlock()
	}

	if err != nil {
		snapshot.Destroy(zfs.DestroyDeferDeletion)
		return err
	}
	return snapshot.Destroy(zfs.DestroyDeferDeletion)
}

func (d *Driver) zfsPath(id string) string {
	return d.options.fsName + "/" + id
}

func (d *Driver) mountPath(id string) string {
	return path.Join(d.options.mountPath, "graph", getMountpoint(id))
}

// CreateReadWrite creates a layer that is writable for use as a container
// file system.
func (d *Driver) CreateReadWrite(id, parent string, opts *graphdriver.CreateOpts) error {
	return d.Create(id, parent, opts)
}

// Create prepares the dataset and filesystem for the ZFS driver for the given id under the parent.
func (d *Driver) Create(id, parent string, opts *graphdriver.CreateOpts) error {
	var storageOpt map[string]string
	if opts != nil {
		storageOpt = opts.StorageOpt
	}

	err := d.create(id, parent, storageOpt)
	if err == nil {
		return nil
	}
	if zfsError, ok := err.(*zfs.Error); ok {
		if !strings.HasSuffix(zfsError.Stderr, "dataset already exists\n") {
			return err
		}
		// aborted build -> cleanup
	} else {
		return err
	}

	dataset := zfs.Dataset{Name: d.zfsPath(id)}
	if err := dataset.Destroy(zfs.DestroyRecursiveClones); err != nil {
		return err
	}

	// retry
	return d.create(id, parent, storageOpt)
}

func (d *Driver) create(id, parent string, storageOpt map[string]string) error {
	name := d.zfsPath(id)
	quota, err := parseStorageOpt(storageOpt)
	if err != nil {
		return err
	}
	if parent == "" {
		mountoptions := map[string]string{"mountpoint": "legacy"}
		fs, err := zfs.CreateFilesystem(name, mountoptions)
		if err == nil {
			err = setQuota(name, quota)
			if err == nil {
				d.Lock()
				d.filesystemsCache[fs.Name] = true
				d.Unlock()
			}
		}
		return err
	}
	err = d.cloneFilesystem(name, d.zfsPath(parent))
	if err == nil {
		err = setQuota(name, quota)
	}
	return err
}

func parseStorageOpt(storageOpt map[string]string) (string, error) {
	// Read size to change the disk quota per container
	for k, v := range storageOpt {
		key := strings.ToLower(k)
		switch key {
		case "size":
			return v, nil
		default:
			return "0", fmt.Errorf("Unknown option %s", key)
		}
	}
	return "0", nil
}

func setQuota(name string, quota string) error {
	if quota == "0" {
		return nil
	}
	fs, err := zfs.GetDataset(name)
	if err != nil {
		return err
	}
	return fs.SetProperty("quota", quota)
}

// Remove deletes the dataset, filesystem and the cache for the given id.
func (d *Driver) Remove(id string) error {
	name := d.zfsPath(id)
	dataset := zfs.Dataset{Name: name}
	err := dataset.Destroy(zfs.DestroyRecursive)
	if err == nil {
		d.Lock()
		delete(d.filesystemsCache, name)
		d.Unlock()
	}
	return err
}

// Get returns the mountpoint for the given id after creating the target directories if necessary.
func (d *Driver) Get(id, mountLabel string) (string, error) {
	mountpoint := d.mountPath(id)
	if count := d.ctr.Increment(mountpoint); count > 1 {
		return mountpoint, nil
	}

	filesystem := d.zfsPath(id)
	options := label.FormatMountLabel("", mountLabel)
	logrus.Debugf(`[zfs] mount("%s", "%s", "%s")`, filesystem, mountpoint, options)

	rootUID, rootGID, err := idtools.GetRootUIDGID(d.uidMaps, d.gidMaps)
	if err != nil {
		d.ctr.Decrement(mountpoint)
		return "", err
	}
	// Create the target directories if they don't exist
	if err := idtools.MkdirAllAs(mountpoint, 0755, rootUID, rootGID); err != nil {
		d.ctr.Decrement(mountpoint)
		return "", err
	}

	if err := mount.Mount(filesystem, mountpoint, "zfs", options); err != nil {
		d.ctr.Decrement(mountpoint)
		return "", fmt.Errorf("error creating zfs mount of %s to %s: %v", filesystem, mountpoint, err)
	}

	// this could be our first mount after creation of the filesystem, and the root dir may still have root
	// permissions instead of the remapped root uid:gid (if user namespaces are enabled):
	if err := os.Chown(mountpoint, rootUID, rootGID); err != nil {
		mount.Unmount(mountpoint)
		d.ctr.Decrement(mountpoint)
		return "", fmt.Errorf("error modifying zfs mountpoint (%s) directory ownership: %v", mountpoint, err)
	}

	return mountpoint, nil
}

// Put removes the existing mountpoint for the given id if it exists.
func (d *Driver) Put(id string) error {
	mountpoint := d.mountPath(id)
	if count := d.ctr.Decrement(mountpoint); count > 0 {
		return nil
	}
	mounted, err := graphdriver.Mounted(graphdriver.FsMagicZfs, mountpoint)
	if err != nil || !mounted {
		return err
	}

	logrus.Debugf(`[zfs] unmount("%s")`, mountpoint)

	if err := mount.Unmount(mountpoint); err != nil {
		return fmt.Errorf("error unmounting to %s: %v", mountpoint, err)
	}
	return nil
}

// Exists checks to see if the cache entry exists for the given id.
func (d *Driver) Exists(id string) bool {
	d.Lock()
	defer d.Unlock()
	return d.filesystemsCache[d.zfsPath(id)] == true
}
