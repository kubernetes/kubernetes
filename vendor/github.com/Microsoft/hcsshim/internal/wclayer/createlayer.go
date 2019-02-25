package wclayer

import (
	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/sirupsen/logrus"
)

// CreateLayer creates a new, empty, read-only layer on the filesystem based on
// the parent layer provided.
func CreateLayer(path, parent string) (err error) {
	title := "hcsshim::CreateLayer"
	fields := logrus.Fields{
		"parent": parent,
		"path":   path,
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

	err = createLayer(&stdDriverInfo, path, parent)
	if err != nil {
		return hcserror.New(err, title+" - failed", "")
	}
	return nil
}
