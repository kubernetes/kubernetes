package graphdriver

import (
	"io"
	"time"

	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/chrootarchive"
	"github.com/docker/docker/pkg/idtools"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/sirupsen/logrus"
)

var (
	// ApplyUncompressedLayer defines the unpack method used by the graph
	// driver.
	ApplyUncompressedLayer = chrootarchive.ApplyUncompressedLayer
)

// NaiveDiffDriver takes a ProtoDriver and adds the
// capability of the Diffing methods on the local file system,
// which it may or may not support on its own. See the comment
// on the exported NewNaiveDiffDriver function below.
// Notably, the AUFS driver doesn't need to be wrapped like this.
type NaiveDiffDriver struct {
	ProtoDriver
	uidMaps []idtools.IDMap
	gidMaps []idtools.IDMap
}

// NewNaiveDiffDriver returns a fully functional driver that wraps the
// given ProtoDriver and adds the capability of the following methods which
// it may or may not support on its own:
//     Diff(id, parent string) (archive.Archive, error)
//     Changes(id, parent string) ([]archive.Change, error)
//     ApplyDiff(id, parent string, diff archive.Reader) (size int64, err error)
//     DiffSize(id, parent string) (size int64, err error)
func NewNaiveDiffDriver(driver ProtoDriver, uidMaps, gidMaps []idtools.IDMap) Driver {
	return &NaiveDiffDriver{ProtoDriver: driver,
		uidMaps: uidMaps,
		gidMaps: gidMaps}
}

// Diff produces an archive of the changes between the specified
// layer and its parent layer which may be "".
func (gdw *NaiveDiffDriver) Diff(id, parent string) (arch io.ReadCloser, err error) {
	startTime := time.Now()
	driver := gdw.ProtoDriver

	layerRootFs, err := driver.Get(id, "")
	if err != nil {
		return nil, err
	}
	layerFs := layerRootFs.Path()

	defer func() {
		if err != nil {
			driver.Put(id)
		}
	}()

	if parent == "" {
		archive, err := archive.Tar(layerFs, archive.Uncompressed)
		if err != nil {
			return nil, err
		}
		return ioutils.NewReadCloserWrapper(archive, func() error {
			err := archive.Close()
			driver.Put(id)
			return err
		}), nil
	}

	parentRootFs, err := driver.Get(parent, "")
	if err != nil {
		return nil, err
	}
	defer driver.Put(parent)

	parentFs := parentRootFs.Path()

	changes, err := archive.ChangesDirs(layerFs, parentFs)
	if err != nil {
		return nil, err
	}

	archive, err := archive.ExportChanges(layerFs, changes, gdw.uidMaps, gdw.gidMaps)
	if err != nil {
		return nil, err
	}

	return ioutils.NewReadCloserWrapper(archive, func() error {
		err := archive.Close()
		driver.Put(id)

		// NaiveDiffDriver compares file metadata with parent layers. Parent layers
		// are extracted from tar's with full second precision on modified time.
		// We need this hack here to make sure calls within same second receive
		// correct result.
		time.Sleep(time.Until(startTime.Truncate(time.Second).Add(time.Second)))
		return err
	}), nil
}

// Changes produces a list of changes between the specified layer
// and its parent layer. If parent is "", then all changes will be ADD changes.
func (gdw *NaiveDiffDriver) Changes(id, parent string) ([]archive.Change, error) {
	driver := gdw.ProtoDriver

	layerRootFs, err := driver.Get(id, "")
	if err != nil {
		return nil, err
	}
	defer driver.Put(id)

	layerFs := layerRootFs.Path()
	parentFs := ""

	if parent != "" {
		parentRootFs, err := driver.Get(parent, "")
		if err != nil {
			return nil, err
		}
		defer driver.Put(parent)
		parentFs = parentRootFs.Path()
	}

	return archive.ChangesDirs(layerFs, parentFs)
}

// ApplyDiff extracts the changeset from the given diff into the
// layer with the specified id and parent, returning the size of the
// new layer in bytes.
func (gdw *NaiveDiffDriver) ApplyDiff(id, parent string, diff io.Reader) (size int64, err error) {
	driver := gdw.ProtoDriver

	// Mount the root filesystem so we can apply the diff/layer.
	layerRootFs, err := driver.Get(id, "")
	if err != nil {
		return
	}
	defer driver.Put(id)

	layerFs := layerRootFs.Path()
	options := &archive.TarOptions{UIDMaps: gdw.uidMaps,
		GIDMaps: gdw.gidMaps}
	start := time.Now().UTC()
	logrus.Debug("Start untar layer")
	if size, err = ApplyUncompressedLayer(layerFs, diff, options); err != nil {
		return
	}
	logrus.Debugf("Untar time: %vs", time.Now().UTC().Sub(start).Seconds())

	return
}

// DiffSize calculates the changes between the specified layer
// and its parent and returns the size in bytes of the changes
// relative to its base filesystem directory.
func (gdw *NaiveDiffDriver) DiffSize(id, parent string) (size int64, err error) {
	driver := gdw.ProtoDriver

	changes, err := gdw.Changes(id, parent)
	if err != nil {
		return
	}

	layerFs, err := driver.Get(id, "")
	if err != nil {
		return
	}
	defer driver.Put(id)

	return archive.ChangesSize(layerFs.Path(), changes), nil
}
