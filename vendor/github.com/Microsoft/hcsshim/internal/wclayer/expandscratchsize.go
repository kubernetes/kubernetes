package wclayer

import (
	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/sirupsen/logrus"
)

// ExpandScratchSize expands the size of a layer to at least size bytes.
func ExpandScratchSize(path string, size uint64) (err error) {
	title := "hcsshim::ExpandScratchSize"
	fields := logrus.Fields{
		"path": path,
		"size": size,
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

	err = expandSandboxSize(&stdDriverInfo, path, size)
	if err != nil {
		return hcserror.New(err, title+" - failed", "")
	}
	return nil
}
