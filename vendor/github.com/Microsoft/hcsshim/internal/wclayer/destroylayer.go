package wclayer

import (
	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/sirupsen/logrus"
)

// DestroyLayer will remove the on-disk files representing the layer with the given
// path, including that layer's containing folder, if any.
func DestroyLayer(path string) (err error) {
	title := "hcsshim::DestroyLayer"
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

	err = destroyLayer(&stdDriverInfo, path)
	if err != nil {
		return hcserror.New(err, title+" - failed", "")
	}
	return nil
}
