package wclayer

import (
	"syscall"

	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/sirupsen/logrus"
)

// GetLayerMountPath will look for a mounted layer with the given path and return
// the path at which that layer can be accessed.  This path may be a volume path
// if the layer is a mounted read-write layer, otherwise it is expected to be the
// folder path at which the layer is stored.
func GetLayerMountPath(path string) (_ string, err error) {
	title := "hcsshim::GetLayerMountPath"
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

	var mountPathLength uintptr
	mountPathLength = 0

	// Call the procedure itself.
	logrus.WithFields(fields).Debug("Calling proc (1)")
	err = getLayerMountPath(&stdDriverInfo, path, &mountPathLength, nil)
	if err != nil {
		return "", hcserror.New(err, title+" - failed", "(first call)")
	}

	// Allocate a mount path of the returned length.
	if mountPathLength == 0 {
		return "", nil
	}
	mountPathp := make([]uint16, mountPathLength)
	mountPathp[0] = 0

	// Call the procedure again
	logrus.WithFields(fields).Debug("Calling proc (2)")
	err = getLayerMountPath(&stdDriverInfo, path, &mountPathLength, &mountPathp[0])
	if err != nil {
		return "", hcserror.New(err, title+" - failed", "(second call)")
	}

	mountPath := syscall.UTF16ToString(mountPathp[0:])
	fields["mountPath"] = mountPath
	return mountPath, nil
}
