// +build linux

package overlay

import (
	"bufio"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"strconv"

	"github.com/docker/docker/daemon/graphdriver"
	"github.com/docker/docker/daemon/graphdriver/overlayutils"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/fsutils"
	"github.com/docker/docker/pkg/idtools"
	"github.com/docker/docker/pkg/locker"
	"github.com/docker/docker/pkg/mount"
	"github.com/docker/docker/pkg/system"
	"github.com/opencontainers/selinux/go-selinux/label"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

// This is a small wrapper over the NaiveDiffWriter that lets us have a custom
// implementation of ApplyDiff()

var (
	// ErrApplyDiffFallback is returned to indicate that a normal ApplyDiff is applied as a fallback from Naive diff writer.
	ErrApplyDiffFallback = fmt.Errorf("Fall back to normal ApplyDiff")
	backingFs            = "<unknown>"
)

// ApplyDiffProtoDriver wraps the ProtoDriver by extending the interface with ApplyDiff method.
type ApplyDiffProtoDriver interface {
	graphdriver.ProtoDriver
	// ApplyDiff writes the diff to the archive for the given id and parent id.
	// It returns the size in bytes written if successful, an error ErrApplyDiffFallback is returned otherwise.
	ApplyDiff(id, parent string, diff io.Reader) (size int64, err error)
}

type naiveDiffDriverWithApply struct {
	graphdriver.Driver
	applyDiff ApplyDiffProtoDriver
}

// NaiveDiffDriverWithApply returns a NaiveDiff driver with custom ApplyDiff.
func NaiveDiffDriverWithApply(driver ApplyDiffProtoDriver, uidMaps, gidMaps []idtools.IDMap) graphdriver.Driver {
	return &naiveDiffDriverWithApply{
		Driver:    graphdriver.NewNaiveDiffDriver(driver, uidMaps, gidMaps),
		applyDiff: driver,
	}
}

// ApplyDiff creates a diff layer with either the NaiveDiffDriver or with a fallback.
func (d *naiveDiffDriverWithApply) ApplyDiff(id, parent string, diff io.Reader) (int64, error) {
	b, err := d.applyDiff.ApplyDiff(id, parent, diff)
	if err == ErrApplyDiffFallback {
		return d.Driver.ApplyDiff(id, parent, diff)
	}
	return b, err
}

// This backend uses the overlay union filesystem for containers
// plus hard link file sharing for images.

// Each container/image can have a "root" subdirectory which is a plain
// filesystem hierarchy, or they can use overlay.

// If they use overlay there is a "upper" directory and a "lower-id"
// file, as well as "merged" and "work" directories. The "upper"
// directory has the upper layer of the overlay, and "lower-id" contains
// the id of the parent whose "root" directory shall be used as the lower
// layer in the overlay. The overlay itself is mounted in the "merged"
// directory, and the "work" dir is needed for overlay to work.

// When an overlay layer is created there are two cases, either the
// parent has a "root" dir, then we start out with an empty "upper"
// directory overlaid on the parents root. This is typically the
// case with the init layer of a container which is based on an image.
// If there is no "root" in the parent, we inherit the lower-id from
// the parent and start by making a copy in the parent's "upper" dir.
// This is typically the case for a container layer which copies
// its parent -init upper layer.

// Additionally we also have a custom implementation of ApplyLayer
// which makes a recursive copy of the parent "root" layer using
// hardlinks to share file data, and then applies the layer on top
// of that. This means all child images share file (but not directory)
// data with the parent.

// Driver contains information about the home directory and the list of active mounts that are created using this driver.
type Driver struct {
	home          string
	uidMaps       []idtools.IDMap
	gidMaps       []idtools.IDMap
	ctr           *graphdriver.RefCounter
	supportsDType bool
	locker        *locker.Locker
}

func init() {
	graphdriver.Register("overlay", Init)
}

// Init returns the NaiveDiffDriver, a native diff driver for overlay filesystem.
// If overlay filesystem is not supported on the host, graphdriver.ErrNotSupported is returned as error.
// If an overlay filesystem is not supported over an existing filesystem then error graphdriver.ErrIncompatibleFS is returned.
func Init(home string, options []string, uidMaps, gidMaps []idtools.IDMap) (graphdriver.Driver, error) {

	if err := supportsOverlay(); err != nil {
		return nil, graphdriver.ErrNotSupported
	}

	fsMagic, err := graphdriver.GetFSMagic(home)
	if err != nil {
		return nil, err
	}
	if fsName, ok := graphdriver.FsNames[fsMagic]; ok {
		backingFs = fsName
	}

	switch fsMagic {
	case graphdriver.FsMagicAufs, graphdriver.FsMagicBtrfs, graphdriver.FsMagicOverlay, graphdriver.FsMagicZfs, graphdriver.FsMagicEcryptfs:
		logrus.Errorf("'overlay' is not supported over %s", backingFs)
		return nil, graphdriver.ErrIncompatibleFS
	}

	rootUID, rootGID, err := idtools.GetRootUIDGID(uidMaps, gidMaps)
	if err != nil {
		return nil, err
	}
	// Create the driver home dir
	if err := idtools.MkdirAllAs(home, 0700, rootUID, rootGID); err != nil && !os.IsExist(err) {
		return nil, err
	}

	if err := mount.MakePrivate(home); err != nil {
		return nil, err
	}

	supportsDType, err := fsutils.SupportsDType(home)
	if err != nil {
		return nil, err
	}
	if !supportsDType {
		// not a fatal error until v17.12 (#27443)
		logrus.Warn(overlayutils.ErrDTypeNotSupported("overlay", backingFs))
	}

	d := &Driver{
		home:          home,
		uidMaps:       uidMaps,
		gidMaps:       gidMaps,
		ctr:           graphdriver.NewRefCounter(graphdriver.NewFsChecker(graphdriver.FsMagicOverlay)),
		supportsDType: supportsDType,
		locker:        locker.New(),
	}

	return NaiveDiffDriverWithApply(d, uidMaps, gidMaps), nil
}

func supportsOverlay() error {
	// We can try to modprobe overlay first before looking at
	// proc/filesystems for when overlay is supported
	exec.Command("modprobe", "overlay").Run()

	f, err := os.Open("/proc/filesystems")
	if err != nil {
		return err
	}
	defer f.Close()

	s := bufio.NewScanner(f)
	for s.Scan() {
		if s.Text() == "nodev\toverlay" {
			return nil
		}
	}
	logrus.Error("'overlay' not found as a supported filesystem on this host. Please ensure kernel is new enough and has overlay support loaded.")
	return graphdriver.ErrNotSupported
}

func (d *Driver) String() string {
	return "overlay"
}

// Status returns current driver information in a two dimensional string array.
// Output contains "Backing Filesystem" used in this implementation.
func (d *Driver) Status() [][2]string {
	return [][2]string{
		{"Backing Filesystem", backingFs},
		{"Supports d_type", strconv.FormatBool(d.supportsDType)},
	}
}

// GetMetadata returns meta data about the overlay driver such as root, LowerDir, UpperDir, WorkDir and MergeDir used to store data.
func (d *Driver) GetMetadata(id string) (map[string]string, error) {
	dir := d.dir(id)
	if _, err := os.Stat(dir); err != nil {
		return nil, err
	}

	metadata := make(map[string]string)

	// If id has a root, it is an image
	rootDir := path.Join(dir, "root")
	if _, err := os.Stat(rootDir); err == nil {
		metadata["RootDir"] = rootDir
		return metadata, nil
	}

	lowerID, err := ioutil.ReadFile(path.Join(dir, "lower-id"))
	if err != nil {
		return nil, err
	}

	metadata["LowerDir"] = path.Join(d.dir(string(lowerID)), "root")
	metadata["UpperDir"] = path.Join(dir, "upper")
	metadata["WorkDir"] = path.Join(dir, "work")
	metadata["MergedDir"] = path.Join(dir, "merged")

	return metadata, nil
}

// Cleanup any state created by overlay which should be cleaned when daemon
// is being shutdown. For now, we just have to unmount the bind mounted
// we had created.
func (d *Driver) Cleanup() error {
	return mount.Unmount(d.home)
}

// CreateReadWrite creates a layer that is writable for use as a container
// file system.
func (d *Driver) CreateReadWrite(id, parent string, opts *graphdriver.CreateOpts) error {
	return d.Create(id, parent, opts)
}

// Create is used to create the upper, lower, and merge directories required for overlay fs for a given id.
// The parent filesystem is used to configure these directories for the overlay.
func (d *Driver) Create(id, parent string, opts *graphdriver.CreateOpts) (retErr error) {

	if opts != nil && len(opts.StorageOpt) != 0 {
		return fmt.Errorf("--storage-opt is not supported for overlay")
	}

	dir := d.dir(id)

	rootUID, rootGID, err := idtools.GetRootUIDGID(d.uidMaps, d.gidMaps)
	if err != nil {
		return err
	}
	if err := idtools.MkdirAllAs(path.Dir(dir), 0700, rootUID, rootGID); err != nil {
		return err
	}
	if err := idtools.MkdirAs(dir, 0700, rootUID, rootGID); err != nil {
		return err
	}

	defer func() {
		// Clean up on failure
		if retErr != nil {
			os.RemoveAll(dir)
		}
	}()

	// Toplevel images are just a "root" dir
	if parent == "" {
		if err := idtools.MkdirAs(path.Join(dir, "root"), 0755, rootUID, rootGID); err != nil {
			return err
		}
		return nil
	}

	parentDir := d.dir(parent)

	// Ensure parent exists
	if _, err := os.Lstat(parentDir); err != nil {
		return err
	}

	// If parent has a root, just do an overlay to it
	parentRoot := path.Join(parentDir, "root")

	if s, err := os.Lstat(parentRoot); err == nil {
		if err := idtools.MkdirAs(path.Join(dir, "upper"), s.Mode(), rootUID, rootGID); err != nil {
			return err
		}
		if err := idtools.MkdirAs(path.Join(dir, "work"), 0700, rootUID, rootGID); err != nil {
			return err
		}
		if err := idtools.MkdirAs(path.Join(dir, "merged"), 0700, rootUID, rootGID); err != nil {
			return err
		}
		if err := ioutil.WriteFile(path.Join(dir, "lower-id"), []byte(parent), 0666); err != nil {
			return err
		}
		return nil
	}

	// Otherwise, copy the upper and the lower-id from the parent

	lowerID, err := ioutil.ReadFile(path.Join(parentDir, "lower-id"))
	if err != nil {
		return err
	}

	if err := ioutil.WriteFile(path.Join(dir, "lower-id"), lowerID, 0666); err != nil {
		return err
	}

	parentUpperDir := path.Join(parentDir, "upper")
	s, err := os.Lstat(parentUpperDir)
	if err != nil {
		return err
	}

	upperDir := path.Join(dir, "upper")
	if err := idtools.MkdirAs(upperDir, s.Mode(), rootUID, rootGID); err != nil {
		return err
	}
	if err := idtools.MkdirAs(path.Join(dir, "work"), 0700, rootUID, rootGID); err != nil {
		return err
	}
	if err := idtools.MkdirAs(path.Join(dir, "merged"), 0700, rootUID, rootGID); err != nil {
		return err
	}

	return copyDir(parentUpperDir, upperDir, 0)
}

func (d *Driver) dir(id string) string {
	return path.Join(d.home, id)
}

// Remove cleans the directories that are created for this id.
func (d *Driver) Remove(id string) error {
	d.locker.Lock(id)
	defer d.locker.Unlock(id)
	return system.EnsureRemoveAll(d.dir(id))
}

// Get creates and mounts the required file system for the given id and returns the mount path.
func (d *Driver) Get(id string, mountLabel string) (s string, err error) {
	d.locker.Lock(id)
	defer d.locker.Unlock(id)
	dir := d.dir(id)
	if _, err := os.Stat(dir); err != nil {
		return "", err
	}
	// If id has a root, just return it
	rootDir := path.Join(dir, "root")
	if _, err := os.Stat(rootDir); err == nil {
		return rootDir, nil
	}
	mergedDir := path.Join(dir, "merged")
	if count := d.ctr.Increment(mergedDir); count > 1 {
		return mergedDir, nil
	}
	defer func() {
		if err != nil {
			if c := d.ctr.Decrement(mergedDir); c <= 0 {
				unix.Unmount(mergedDir, 0)
			}
		}
	}()
	lowerID, err := ioutil.ReadFile(path.Join(dir, "lower-id"))
	if err != nil {
		return "", err
	}
	var (
		lowerDir = path.Join(d.dir(string(lowerID)), "root")
		upperDir = path.Join(dir, "upper")
		workDir  = path.Join(dir, "work")
		opts     = fmt.Sprintf("lowerdir=%s,upperdir=%s,workdir=%s", lowerDir, upperDir, workDir)
	)
	if err := unix.Mount("overlay", mergedDir, "overlay", 0, label.FormatMountLabel(opts, mountLabel)); err != nil {
		return "", fmt.Errorf("error creating overlay mount to %s: %v", mergedDir, err)
	}
	// chown "workdir/work" to the remapped root UID/GID. Overlay fs inside a
	// user namespace requires this to move a directory from lower to upper.
	rootUID, rootGID, err := idtools.GetRootUIDGID(d.uidMaps, d.gidMaps)
	if err != nil {
		return "", err
	}
	if err := os.Chown(path.Join(workDir, "work"), rootUID, rootGID); err != nil {
		return "", err
	}
	return mergedDir, nil
}

// Put unmounts the mount path created for the give id.
func (d *Driver) Put(id string) error {
	d.locker.Lock(id)
	defer d.locker.Unlock(id)
	// If id has a root, just return
	if _, err := os.Stat(path.Join(d.dir(id), "root")); err == nil {
		return nil
	}
	mountpoint := path.Join(d.dir(id), "merged")
	if count := d.ctr.Decrement(mountpoint); count > 0 {
		return nil
	}
	if err := unix.Unmount(mountpoint, unix.MNT_DETACH); err != nil {
		logrus.Debugf("Failed to unmount %s overlay: %v", id, err)
	}
	return nil
}

// ApplyDiff applies the new layer on top of the root, if parent does not exist with will return an ErrApplyDiffFallback error.
func (d *Driver) ApplyDiff(id string, parent string, diff io.Reader) (size int64, err error) {
	dir := d.dir(id)

	if parent == "" {
		return 0, ErrApplyDiffFallback
	}

	parentRootDir := path.Join(d.dir(parent), "root")
	if _, err := os.Stat(parentRootDir); err != nil {
		return 0, ErrApplyDiffFallback
	}

	// We now know there is a parent, and it has a "root" directory containing
	// the full root filesystem. We can just hardlink it and apply the
	// layer. This relies on two things:
	// 1) ApplyDiff is only run once on a clean (no writes to upper layer) container
	// 2) ApplyDiff doesn't do any in-place writes to files (would break hardlinks)
	// These are all currently true and are not expected to break

	tmpRootDir, err := ioutil.TempDir(dir, "tmproot")
	if err != nil {
		return 0, err
	}
	defer func() {
		if err != nil {
			os.RemoveAll(tmpRootDir)
		} else {
			os.RemoveAll(path.Join(dir, "upper"))
			os.RemoveAll(path.Join(dir, "work"))
			os.RemoveAll(path.Join(dir, "merged"))
			os.RemoveAll(path.Join(dir, "lower-id"))
		}
	}()

	if err = copyDir(parentRootDir, tmpRootDir, copyHardlink); err != nil {
		return 0, err
	}

	options := &archive.TarOptions{UIDMaps: d.uidMaps, GIDMaps: d.gidMaps}
	if size, err = graphdriver.ApplyUncompressedLayer(tmpRootDir, diff, options); err != nil {
		return 0, err
	}

	rootDir := path.Join(dir, "root")
	if err := os.Rename(tmpRootDir, rootDir); err != nil {
		return 0, err
	}

	return
}

// Exists checks to see if the id is already mounted.
func (d *Driver) Exists(id string) bool {
	_, err := os.Stat(d.dir(id))
	return err == nil
}
