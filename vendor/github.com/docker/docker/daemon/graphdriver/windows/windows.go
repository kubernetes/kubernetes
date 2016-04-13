//+build windows

package windows

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/daemon/graphdriver"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/chrootarchive"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/microsoft/hcsshim"
)

func init() {
	graphdriver.Register("windowsfilter", InitFilter)
	graphdriver.Register("windowsdiff", InitDiff)
}

const (
	diffDriver = iota
	filterDriver
)

type WindowsGraphDriver struct {
	info       hcsshim.DriverInfo
	sync.Mutex // Protects concurrent modification to active
	active     map[string]int
}

// New returns a new Windows storage filter driver.
func InitFilter(home string, options []string) (graphdriver.Driver, error) {
	logrus.Debugf("WindowsGraphDriver InitFilter at %s", home)
	d := &WindowsGraphDriver{
		info: hcsshim.DriverInfo{
			HomeDir: home,
			Flavour: filterDriver,
		},
		active: make(map[string]int),
	}
	return d, nil
}

// New returns a new Windows differencing disk driver.
func InitDiff(home string, options []string) (graphdriver.Driver, error) {
	logrus.Debugf("WindowsGraphDriver InitDiff at %s", home)
	d := &WindowsGraphDriver{
		info: hcsshim.DriverInfo{
			HomeDir: home,
			Flavour: diffDriver,
		},
		active: make(map[string]int),
	}
	return d, nil
}

func (d *WindowsGraphDriver) Info() hcsshim.DriverInfo {
	return d.info
}

func (d *WindowsGraphDriver) String() string {
	switch d.info.Flavour {
	case diffDriver:
		return "windowsdiff"
	case filterDriver:
		return "windowsfilter"
	default:
		return "Unknown driver flavour"
	}
}

func (d *WindowsGraphDriver) Status() [][2]string {
	return [][2]string{
		{"Windows", ""},
	}
}

// Exists returns true if the given id is registered with
// this driver
func (d *WindowsGraphDriver) Exists(id string) bool {
	result, err := hcsshim.LayerExists(d.info, id)
	if err != nil {
		return false
	}
	return result
}

func (d *WindowsGraphDriver) Create(id, parent string) error {
	return hcsshim.CreateLayer(d.info, id, parent)
}

func (d *WindowsGraphDriver) dir(id string) string {
	return filepath.Join(d.info.HomeDir, filepath.Base(id))
}

// Remove unmounts and removes the dir information
func (d *WindowsGraphDriver) Remove(id string) error {
	return hcsshim.DestroyLayer(d.info, id)
}

// Get returns the rootfs path for the id. This will mount the dir at it's given path
func (d *WindowsGraphDriver) Get(id, mountLabel string) (string, error) {
	var dir string

	d.Lock()
	defer d.Unlock()

	if d.active[id] == 0 {
		if err := hcsshim.ActivateLayer(d.info, id); err != nil {
			return "", err
		}
	}

	mountPath, err := hcsshim.GetLayerMountPath(d.info, id)
	if err != nil {
		return "", err
	}

	// If the layer has a mount path, use that. Otherwise, use the
	// folder path.
	if mountPath != "" {
		dir = mountPath
	} else {
		dir = d.dir(id)
	}

	d.active[id]++

	return dir, nil
}

func (d *WindowsGraphDriver) Put(id string) error {
	logrus.Debugf("WindowsGraphDriver Put() id %s", id)

	d.Lock()
	defer d.Unlock()

	if d.active[id] > 1 {
		d.active[id]--
	} else if d.active[id] == 1 {
		if err := hcsshim.DeactivateLayer(d.info, id); err != nil {
			return err
		}
		delete(d.active, id)
	}

	return nil
}

func (d *WindowsGraphDriver) Cleanup() error {
	return nil
}

// Diff produces an archive of the changes between the specified
// layer and its parent layer which may be "".
func (d *WindowsGraphDriver) Diff(id, parent string) (arch archive.Archive, err error) {
	return nil, fmt.Errorf("The Windows graphdriver does not support Diff()")
}

// Changes produces a list of changes between the specified layer
// and its parent layer. If parent is "", then all changes will be ADD changes.
func (d *WindowsGraphDriver) Changes(id, parent string) ([]archive.Change, error) {
	return nil, fmt.Errorf("The Windows graphdriver does not support Changes()")
}

// ApplyDiff extracts the changeset from the given diff into the
// layer with the specified id and parent, returning the size of the
// new layer in bytes.
func (d *WindowsGraphDriver) ApplyDiff(id, parent string, diff archive.ArchiveReader) (size int64, err error) {
	start := time.Now().UTC()
	logrus.Debugf("WindowsGraphDriver ApplyDiff: Start untar layer")

	destination := d.dir(id)
	if d.info.Flavour == diffDriver {
		destination = filepath.Dir(destination)
	}

	if size, err = chrootarchive.ApplyLayer(destination, diff); err != nil {
		return
	}
	logrus.Debugf("WindowsGraphDriver ApplyDiff: Untar time: %vs", time.Now().UTC().Sub(start).Seconds())

	return
}

// DiffSize calculates the changes between the specified layer
// and its parent and returns the size in bytes of the changes
// relative to its base filesystem directory.
func (d *WindowsGraphDriver) DiffSize(id, parent string) (size int64, err error) {
	changes, err := d.Changes(id, parent)
	if err != nil {
		return
	}

	layerFs, err := d.Get(id, "")
	if err != nil {
		return
	}
	defer d.Put(id)

	return archive.ChangesSize(layerFs, changes), nil
}

func (d *WindowsGraphDriver) CopyDiff(sourceId, id string, parentLayerPaths []string) error {
	d.Lock()
	defer d.Unlock()

	if d.info.Flavour == filterDriver && d.active[sourceId] == 0 {
		if err := hcsshim.ActivateLayer(d.info, sourceId); err != nil {
			return err
		}
		defer func() {
			err := hcsshim.DeactivateLayer(d.info, sourceId)
			if err != nil {
				logrus.Warnf("Failed to Deactivate %s: %s", sourceId, err)
			}
		}()
	}

	return hcsshim.CopyLayer(d.info, sourceId, id, parentLayerPaths)
}

func (d *WindowsGraphDriver) LayerIdsToPaths(ids []string) []string {
	var paths []string
	for _, id := range ids {
		path, err := d.Get(id, "")
		if err != nil {
			logrus.Debug("LayerIdsToPaths: Error getting mount path for id", id, ":", err.Error())
			return nil
		}
		if d.Put(id) != nil {
			logrus.Debug("LayerIdsToPaths: Error putting mount path for id", id, ":", err.Error())
			return nil
		}
		paths = append(paths, path)
	}
	return paths
}

func (d *WindowsGraphDriver) GetMetadata(id string) (map[string]string, error) {
	return nil, nil
}

func (d *WindowsGraphDriver) Export(id string, parentLayerPaths []string) (arch archive.Archive, err error) {
	layerFs, err := d.Get(id, "")
	if err != nil {
		return
	}
	defer func() {
		if err != nil {
			d.Put(id)
		}
	}()

	tempFolder := layerFs + "-temp"
	if err = os.MkdirAll(tempFolder, 0755); err != nil {
		logrus.Errorf("Could not create %s %s", tempFolder, err)
		return
	}
	defer func() {
		if err != nil {
			if err2 := os.RemoveAll(tempFolder); err2 != nil {
				logrus.Warnf("Couldn't clean-up tempFolder: %s %s", tempFolder, err2)
			}
		}
	}()

	if err = hcsshim.ExportLayer(d.info, id, tempFolder, parentLayerPaths); err != nil {
		return
	}

	archive, err := archive.Tar(tempFolder, archive.Uncompressed)
	if err != nil {
		return
	}
	return ioutils.NewReadCloserWrapper(archive, func() error {
		err := archive.Close()
		d.Put(id)
		if err2 := os.RemoveAll(tempFolder); err2 != nil {
			logrus.Warnf("Couldn't clean-up tempFolder: %s %s", tempFolder, err2)
		}
		return err
	}), nil

}

func (d *WindowsGraphDriver) Import(id string, layerData archive.ArchiveReader, parentLayerPaths []string) (size int64, err error) {
	layerFs, err := d.Get(id, "")
	if err != nil {
		return
	}
	defer func() {
		if err != nil {
			d.Put(id)
		}
	}()

	tempFolder := layerFs + "-temp"
	if err = os.MkdirAll(tempFolder, 0755); err != nil {
		logrus.Errorf("Could not create %s %s", tempFolder, err)
		return
	}
	defer func() {
		if err2 := os.RemoveAll(tempFolder); err2 != nil {
			logrus.Warnf("Couldn't clean-up tempFolder: %s %s", tempFolder, err2)
		}
	}()

	start := time.Now().UTC()
	logrus.Debugf("Start untar layer")
	if size, err = chrootarchive.ApplyLayer(tempFolder, layerData); err != nil {
		return
	}
	logrus.Debugf("Untar time: %vs", time.Now().UTC().Sub(start).Seconds())

	if err = hcsshim.ImportLayer(d.info, id, tempFolder, parentLayerPaths); err != nil {
		return
	}

	return
}
