package wclayer

import (
	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/sirupsen/logrus"
)

// ActivateLayer will find the layer with the given id and mount it's filesystem.
// For a read/write layer, the mounted filesystem will appear as a volume on the
// host, while a read-only layer is generally expected to be a no-op.
// An activated layer must later be deactivated via DeactivateLayer.
func ActivateLayer(path string) (err error) {
	title := "hcsshim::ActivateLayer"
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

	err = activateLayer(&stdDriverInfo, path)
	if err != nil {
		return hcserror.New(err, title+" - failed", "")
	}
	return nil
}
