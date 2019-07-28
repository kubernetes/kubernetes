package wclayer

import (
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/Microsoft/go-winio"
	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/Microsoft/hcsshim/internal/safefile"
	"github.com/sirupsen/logrus"
)

// ImportLayer will take the contents of the folder at importFolderPath and import
// that into a layer with the id layerId.  Note that in order to correctly populate
// the layer and interperet the transport format, all parent layers must already
// be present on the system at the paths provided in parentLayerPaths.
func ImportLayer(path string, importFolderPath string, parentLayerPaths []string) (err error) {
	title := "hcsshim::ImportLayer"
	fields := logrus.Fields{
		"path":             path,
		"importFolderPath": importFolderPath,
	}
	logrus.WithFields(fields).Debug(title)
	defer func() {
		if err != nil {
			fields[logrus.ErrorKey] = err
			logrus.WithFields(fields).Error(err)
		} else {
			logrus.WithFields(fields).Debug(title + " - succeeded")
		}
	}()

	// Generate layer descriptors
	layers, err := layerPathsToDescriptors(parentLayerPaths)
	if err != nil {
		return err
	}

	err = importLayer(&stdDriverInfo, path, importFolderPath, layers)
	if err != nil {
		return hcserror.New(err, title+" - failed", "")
	}
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

type legacyLayerWriterWrapper struct {
	*legacyLayerWriter
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

	if err = ImportLayer(r.destRoot.Name(), r.path, r.parentLayerPaths); err != nil {
		return err
	}
	for _, name := range r.Tombstones {
		if err = safefile.RemoveRelative(name, r.destRoot); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	// Add any hard links that were collected.
	for _, lnk := range r.PendingLinks {
		if err = safefile.RemoveRelative(lnk.Path, r.destRoot); err != nil && !os.IsNotExist(err) {
			return err
		}
		if err = safefile.LinkRelative(lnk.Target, lnk.TargetRoot, lnk.Path, r.destRoot); err != nil {
			return err
		}
	}
	// Prepare the utility VM for use if one is present in the layer.
	if r.HasUtilityVM {
		err := safefile.EnsureNotReparsePointRelative("UtilityVM", r.destRoot)
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
func NewLayerWriter(path string, parentLayerPaths []string) (LayerWriter, error) {
	if len(parentLayerPaths) == 0 {
		// This is a base layer. It gets imported differently.
		f, err := safefile.OpenRoot(path)
		if err != nil {
			return nil, err
		}
		return &baseLayerWriter{
			root: f,
		}, nil
	}

	importPath, err := ioutil.TempDir("", "hcs")
	if err != nil {
		return nil, err
	}
	w, err := newLegacyLayerWriter(importPath, parentLayerPaths, path)
	if err != nil {
		return nil, err
	}
	return &legacyLayerWriterWrapper{
		legacyLayerWriter: w,
		path:              importPath,
		parentLayerPaths:  parentLayerPaths,
	}, nil
}
