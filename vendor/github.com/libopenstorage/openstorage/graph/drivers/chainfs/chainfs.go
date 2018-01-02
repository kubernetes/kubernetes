// +build linux,have_chainfs

package chainfs

/*
extern int start_chainfs(int mode, char *mount_path);
extern void stop_chainfs(void);
extern int create_layer(char *id, char *parent_id);
extern int remove_layer(char *id);
extern int check_layer(char *id);
extern int alloc_chainfs(char *id);
extern int release_chainfs(char *id);
extern int release_cache(void);
#cgo LDFLAGS: -lfuse -lulockmgr -lchainfs
#cgo CFLAGS: -g3
*/
import "C"

import (
	"io/ioutil"
	"os"
	"path"

	"go.pedge.io/dlog"

	"github.com/libopenstorage/openstorage/api"
	"github.com/libopenstorage/openstorage/graph"

	"github.com/docker/docker/daemon/graphdriver"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/chrootarchive"
	"github.com/docker/docker/pkg/directory"
	"github.com/docker/docker/pkg/idtools"
)

const (
	Name     = "chainfs"
	Type     = api.DriverType_DRIVER_TYPE_GRAPH
	Type     = api.Graph
	virtPath = "/var/lib/openstorage/chainfs"
)

type Driver struct {
}

func Init(home string, options []string, uidMaps, gidMaps []idtools.IDMap) (graphdriver.Driver, error) {
	dlog.Infof("Initializing libchainfs at home: %s and storage: %v...", home, virtPath)

	cVirtPath := C.CString(virtPath)
	go C.start_chainfs(1, cVirtPath)

	d := &Driver{}

	return d, nil
}

func (d *Driver) String() string {
	return "openstorage-chainfs"
}

// Cleanup performs necessary tasks to release resources
// held by the driver, e.g., unmounting all layered filesystems
// known to this driver.
func (d *Driver) Cleanup() error {
	dlog.Infof("Stopping libchainfs at %s", virtPath)
	C.stop_chainfs()
	return nil
}

// Status returns a set of key-value pairs which give low
// level diagnostic status about this driver.
func (d *Driver) Status() [][2]string {
	return [][2]string{
		{"OpenStorage ChainFS", "OK"},
	}
}

// Create creates a new, empty, filesystem layer with the
// specified id and parent and mountLabel. Parent and mountLabel may be "".
func (d *Driver) Create(id string, parent string, ml string, storageOpts map[string]string) error {
	if parent != "" {
		dlog.Infof("Creating layer %s with parent %s", id, parent)
	} else {
		dlog.Infof("Creating parent layer %s", id)
	}

	cID := C.CString(id)
	cParent := C.CString(parent)

	ret, err := C.create_layer(cID, cParent)
	if int(ret) != 0 {
		dlog.Warnf("Error while creating layer %s", id)
		return err
	}

	return nil
}

// Remove attempts to remove the filesystem layer with this id.
func (d *Driver) Remove(id string) error {
	dlog.Infof("Removing layer %s", id)

	cID := C.CString(id)
	ret, err := C.remove_layer(cID)
	if int(ret) != 0 {
		dlog.Warnf("Error while removing layer %s", id)
		return err
	}

	return nil
}

// Returns a set of key-value pairs which give low level information
// about the image/container driver is managing.
func (d *Driver) GetMetadata(id string) (map[string]string, error) {
	return nil, nil
}

// Get returns the mountpoint for the layered filesystem referred
// to by this id. You can optionally specify a mountLabel or "".
// Returns the absolute path to the mounted layered filesystem.
func (d *Driver) Get(id, mountLabel string) (string, error) {
	cID := C.CString(id)

	ret, err := C.alloc_chainfs(cID)
	if int(ret) != 0 {
		dlog.Warnf("Error while creating a chain FS for %s", id)
		return "", err
	} else {
		dlog.Debugf("Created a chain FS for %s", id)
		chainPath := path.Join(virtPath, id)
		return chainPath, err
	}
}

// Put releases the system resources for the specified id,
// e.g, unmounting layered filesystem.
func (d *Driver) Put(id string) error {
	dlog.Debugf("Releasing chain FS for %s", id)

	cID := C.CString(id)
	_, err := C.release_chainfs(cID)

	// C.release_cache()

	return err
}

// Exists returns whether a filesystem layer with the specified
// ID exists on this driver.
// All cache entries exist.
func (d *Driver) Exists(id string) bool {
	path := path.Join(virtPath, id)

	_, err := os.Stat(path)

	if err == nil {
		return true
	} else {
		return false
	}
}

// ApplyDiff extracts the changeset from the given diff into the
// layer with the specified id and parent, returning the size of the
// new layer in bytes.
// The archive.Reader must be an uncompressed stream.
func (d *Driver) ApplyDiff(id string, parent string, diff archive.Reader) (size int64, err error) {
	dir := path.Join(virtPath, id)
	// dir := path.Join("/tmp/chainfs/", id)

	dlog.Infof("Applying diff at path %s\n", dir)

	if err := chrootarchive.UntarUncompressed(diff, dir, nil); err != nil {
		dlog.Warnf("Error while applying diff to %s: %v", id, err)
		return 0, err
	}

	// show invalid whiteouts warning.
	files, err := ioutil.ReadDir(path.Join(dir, archive.WhiteoutLinkDir))
	if err == nil && len(files) > 0 {
		dlog.Warnf("Archive contains aufs hardlink references that are not supported.")
	}

	return d.DiffSize(id, parent)
}

// Changes produces a list of changes between the specified layer
// and its parent layer. If parent is "", then all changes will be ADD changes.
func (d *Driver) Changes(id, parent string) ([]archive.Change, error) {

	return nil, nil
}

// Diff produces an archive of the changes between the specified
// layer and its parent layer which may be "".
func (d *Driver) Diff(id, parent string) (archive.Archive, error) {
	return archive.TarWithOptions(path.Join(virtPath, id), &archive.TarOptions{
		Compression:     archive.Uncompressed,
		ExcludePatterns: []string{archive.WhiteoutMetaPrefix + "*", "!" + archive.WhiteoutOpaqueDir},
	})
}

// DiffSize calculates the changes between the specified id
// and its parent and returns the size in bytes of the changes
// relative to its base filesystem directory.
func (d *Driver) DiffSize(id, parent string) (size int64, err error) {
	return directory.Size(path.Join(virtPath, id))
}

func init() {
	graph.Register(Name, Init)
}
