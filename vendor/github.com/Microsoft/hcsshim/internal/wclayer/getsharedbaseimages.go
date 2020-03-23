package wclayer

import (
	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/Microsoft/hcsshim/internal/interop"
	"github.com/sirupsen/logrus"
)

// GetSharedBaseImages will enumerate the images stored in the common central
// image store and return descriptive info about those images for the purpose
// of registering them with the graphdriver, graph, and tagstore.
func GetSharedBaseImages() (imageData string, err error) {
	title := "hcsshim::GetSharedBaseImages"
	logrus.Debug(title)
	defer func() {
		if err != nil {
			logrus.WithError(err).Error(err)
		} else {
			logrus.WithField("imageData", imageData).Debug(title + " - succeeded")
		}
	}()

	var buffer *uint16
	err = getBaseImages(&buffer)
	if err != nil {
		return "", hcserror.New(err, title+" - failed", "")
	}
	return interop.ConvertAndFreeCoTaskMemString(buffer), nil
}
