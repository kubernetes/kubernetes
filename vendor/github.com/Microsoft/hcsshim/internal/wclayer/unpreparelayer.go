package wclayer

import (
	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/sirupsen/logrus"
)

// UnprepareLayer disables the filesystem filter for the read-write layer with
// the given id.
func UnprepareLayer(path string) (err error) {
	title := "hcsshim::UnprepareLayer"
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

	err = unprepareLayer(&stdDriverInfo, path)
	if err != nil {
		return hcserror.New(err, title+" - failed", "")
	}
	return nil
}
