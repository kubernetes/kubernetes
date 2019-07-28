package wclayer

import (
	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/sirupsen/logrus"
)

// LayerExists will return true if a layer with the given id exists and is known
// to the system.
func LayerExists(path string) (_ bool, err error) {
	title := "hcsshim::LayerExists"
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

	// Call the procedure itself.
	var exists uint32
	err = layerExists(&stdDriverInfo, path, &exists)
	if err != nil {
		return false, hcserror.New(err, title+" - failed", "")
	}
	fields["layer-exists"] = exists != 0
	return exists != 0, nil
}
