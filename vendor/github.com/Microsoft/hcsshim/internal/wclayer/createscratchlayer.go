package wclayer

import (
	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/sirupsen/logrus"
)

// CreateScratchLayer creates and populates new read-write layer for use by a container.
// This requires both the id of the direct parent layer, as well as the full list
// of paths to all parent layers up to the base (and including the direct parent
// whose id was provided).
func CreateScratchLayer(path string, parentLayerPaths []string) (err error) {
	title := "hcsshim::CreateScratchLayer"
	fields := logrus.Fields{
		"path": path,
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

	err = createSandboxLayer(&stdDriverInfo, path, 0, layers)
	if err != nil {
		return hcserror.New(err, title+" - failed", "")
	}
	return nil
}
