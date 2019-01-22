package hcsshim

import (
	"errors"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/Microsoft/go-winio"
	"github.com/sirupsen/logrus"
)

// ImportLayer will take the contents of the folder at importFolderPath and import
// that into a layer with the id layerId.  Note that in order to correctly populate
// the layer and interperet the transport format, all parent layers must already
// be present on the system at the paths provided in parentLayerPaths.
func ImportLayer(info DriverInfo, layerID string, importFolderPath string, parentLayerPaths []string) error {
	title := "hcsshim::ImportLayer "
	logrus.Debugf(title+"flavour %d layerId %s folder %s", info.Flavour, layerID, importFolderPath)

	// Generate layer descriptors
	layers, err := layerPathsToDescriptors(parentLayerPaths)
	if err != nil {
		return err
	}

	// Convert info to API calling convention
	infop, err := convertDriverInfo(info)
	if err != nil {
		logrus.Error(err)
		return err
	}

	err = importLayer(&infop, layerID, importFolderPath, layers)
	if err != nil {
		err = makeErrorf(err, title, "layerId=%s flavour=%d folder=%s", layerID, info.Flavour, importFolderPath)
		logrus.Error(err)
		return err
	}

	logrus.Debugf(title+"succeeded flavour=%d layerId=%s folder=%s", info.Flavour, layerID, importFolderPath)
	return nil
}

// LayerWriter is an interface that supports writing a new container image layer.
type LayerWriter interface {
	// Add adds a file to the layer with given metadata.
	Add(name string, fileInfo *winio.FileBasicInfo) error
	// AddLink adds a hard link to the layer. The target must already have been added.
	AddLink(name string, target string) error
	// Remove removes a file that was present in a parent layer from the layer.
	Remove(name string) error
	// Write writes data to the current file. The data must be in the format of a Win32
	// backup stream.
	Write(b []byte) (int, error)
	// Close finishes the layer writing process and releases any resources.
	Close() error
}

// FilterLayerWriter provides an interface to write the contents of a layer to the file system.
type FilterLayerWriter struct {
	context uintptr
}

// Add adds a file or directory to the layer. The file's parent directory must have already been added.
//
// name contains the file's relative path. fileInfo contains file times and file attributes; the rest
// of the file metadata and the file data must be written as a Win32 backup stream to the Write() method.
// winio.BackupStreamWriter can be used to facilitate this.
func (w *FilterLayerWriter) Add(name string, fileInfo *winio.FileBasicInfo) error {
	if name[0] != '\\' {
		name = `\` + name
	}
	err := importLayerNext(w.context, name, fileInfo)
	if err != nil {
		return makeError(err, "ImportLayerNext", "")
	}
	return nil
}

// AddLink adds a hard link to the layer. The target of the link must have already been added.
func (w *FilterLayerWriter) AddLink(name string, target string) error {
	return errors.New("hard links not yet supported")
}

// Remove removes a file from the layer. The file must have been present in the parent layer.
//
// name contains the file's relative path.
func (w *FilterLayerWriter) Remove(name string) error {
	if name[0] != '\\' {
		name = `\` + name
	}
	err := importLayerNext(w.context, name, nil)
	if err != nil {
		return makeError(err, "ImportLayerNext", "")
	}
	return nil
}

// Write writes more backup stream data to the current file.
func (w *FilterLayerWriter) Write(b []byte) (int, error) {
	err := importLayerWrite(w.context, b)
	if err != nil {
		err = makeError(err, "ImportLayerWrite", "")
		return 0, err
	}
	return len(b), err
}

// Close completes the layer write operation. The error must be checked to ensure that the
// operation was successful.
func (w *FilterLayerWriter) Close() (err error) {
	if w.context != 0 {
		err = importLayerEnd(w.context)
		if err != nil {
			err = makeError(err, "ImportLayerEnd", "")
		}
		w.context = 0
	}
	return
}

type legacyLayerWriterWrapper struct {
	*legacyLayerWriter
	info             DriverInfo
	layerID          string
	path             string
	parentLayerPaths []string
}

func (r *legacyLayerWriterWrapper) Close() error {
	defer os.RemoveAll(r.root.Name())
	defer r.legacyLayerWriter.CloseRoots()
	err := r.legacyLayerWriter.Close()
	if err != nil {
		return err
	}

	info := r.info
	info.HomeDir = ""
	if err = ImportLayer(info, r.destRoot.Name(), r.path, r.parentLayerPaths); err != nil {
		return err
	}
	for _, name := range r.Tombstones {
		if err = removeRelative(name, r.destRoot); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	// Add any hard links that were collected.
	for _, lnk := range r.PendingLinks {
		if err = removeRelative(lnk.Path, r.destRoot); err != nil && !os.IsNotExist(err) {
			return err
		}
		if err = linkRelative(lnk.Target, lnk.TargetRoot, lnk.Path, r.destRoot); err != nil {
			return err
		}
	}
	// Prepare the utility VM for use if one is present in the layer.
	if r.HasUtilityVM {
		err := ensureNotReparsePointRelative("UtilityVM", r.destRoot)
		if err != nil {
			return err
		}
		err = ProcessUtilityVMImage(filepath.Join(r.destRoot.Name(), "UtilityVM"))
		if err != nil {
			return err
		}
	}
	return nil
}

// NewLayerWriter returns a new layer writer for creating a layer on disk.
// The caller must have taken the SeBackupPrivilege and SeRestorePrivilege privileges
// to call this and any methods on the resulting LayerWriter.
func NewLayerWriter(info DriverInfo, layerID string, parentLayerPaths []string) (LayerWriter, error) {
	if len(parentLayerPaths) == 0 {
		// This is a base layer. It gets imported differently.
		f, err := openRoot(filepath.Join(info.HomeDir, layerID))
		if err != nil {
			return nil, err
		}
		return &baseLayerWriter{
			root: f,
		}, nil
	}

	if procImportLayerBegin.Find() != nil {
		// The new layer reader is not available on this Windows build. Fall back to the
		// legacy export code path.
		path, err := ioutil.TempDir("", "hcs")
		if err != nil {
			return nil, err
		}
		w, err := newLegacyLayerWriter(path, parentLayerPaths, filepath.Join(info.HomeDir, layerID))
		if err != nil {
			return nil, err
		}
		return &legacyLayerWriterWrapper{
			legacyLayerWriter: w,
			info:              info,
			layerID:           layerID,
			path:              path,
			parentLayerPaths:  parentLayerPaths,
		}, nil
	}
	layers, err := layerPathsToDescriptors(parentLayerPaths)
	if err != nil {
		return nil, err
	}

	infop, err := convertDriverInfo(info)
	if err != nil {
		return nil, err
	}

	w := &FilterLayerWriter{}
	err = importLayerBegin(&infop, layerID, layers, &w.context)
	if err != nil {
		return nil, makeError(err, "ImportLayerStart", "")
	}
	return w, nil
}
