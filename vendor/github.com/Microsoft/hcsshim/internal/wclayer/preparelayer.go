package wclayer

import (
	"sync"

	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/sirupsen/logrus"
)

var prepareLayerLock sync.Mutex

// PrepareLayer finds a mounted read-write layer matching path and enables the
// the filesystem filter for use on that layer.  This requires the paths to all
// parent layers, and is necessary in order to view or interact with the layer
// as an actual filesystem (reading and writing files, creating directories, etc).
// Disabling the filter must be done via UnprepareLayer.
func PrepareLayer(path string, parentLayerPaths []string) (err error) {
	title := "hcsshim::PrepareLayer"
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

	// This lock is a temporary workaround for a Windows bug. Only allowing one
	// call to prepareLayer at a time vastly reduces the chance of a timeout.
	prepareLayerLock.Lock()
	defer prepareLayerLock.Unlock()
	err = prepareLayer(&stdDriverInfo, path, layers)
	if err != nil {
		return hcserror.New(err, title+" - failed", "")
	}
	return nil
}
