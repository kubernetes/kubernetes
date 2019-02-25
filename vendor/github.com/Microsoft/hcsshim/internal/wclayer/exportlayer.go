package wclayer

import (
	"io/ioutil"
	"os"

	"github.com/Microsoft/go-winio"
	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/sirupsen/logrus"
)

// ExportLayer will create a folder at exportFolderPath and fill that folder with
// the transport format version of the layer identified by layerId. This transport
// format includes any metadata required for later importing the layer (using
// ImportLayer), and requires the full list of parent layer paths in order to
// perform the export.
func ExportLayer(path string, exportFolderPath string, parentLayerPaths []string) (err error) {
	title := "hcsshim::ExportLayer"
	fields := logrus.Fields{
		"path":             path,
		"exportFolderPath": exportFolderPath,
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

	err = exportLayer(&stdDriverInfo, path, exportFolderPath, layers)
	if err != nil {
		return hcserror.New(err, title+" - failed", "")
	}
	return nil
}

type LayerReader interface {
	Next() (string, int64, *winio.FileBasicInfo, error)
	Read(b []byte) (int, error)
	Close() error
}

// NewLayerReader returns a new layer reader for reading the contents of an on-disk layer.
// The caller must have taken the SeBackupPrivilege privilege
// to call this and any methods on the resulting LayerReader.
func NewLayerReader(path string, parentLayerPaths []string) (LayerReader, error) {
	exportPath, err := ioutil.TempDir("", "hcs")
	if err != nil {
		return nil, err
	}
	err = ExportLayer(path, exportPath, parentLayerPaths)
	if err != nil {
		os.RemoveAll(exportPath)
		return nil, err
	}
	return &legacyLayerReaderWrapper{newLegacyLayerReader(exportPath)}, nil
}

type legacyLayerReaderWrapper struct {
	*legacyLayerReader
}

func (r *legacyLayerReaderWrapper) Close() error {
	err := r.legacyLayerReader.Close()
	os.RemoveAll(r.root)
	return err
}
