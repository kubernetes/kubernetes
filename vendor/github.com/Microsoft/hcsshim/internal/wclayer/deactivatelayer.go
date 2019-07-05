package wclayer

import (
	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/sirupsen/logrus"
)

// DeactivateLayer will dismount a layer that was mounted via ActivateLayer.
func DeactivateLayer(path string) (err error) {
	title := "hcsshim::DeactivateLayer"
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

	err = deactivateLayer(&stdDriverInfo, path)
	if err != nil {
		return hcserror.New(err, title+"- failed", "")
	}
	return nil
}
