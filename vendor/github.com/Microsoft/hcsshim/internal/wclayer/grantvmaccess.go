package wclayer

import (
	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/sirupsen/logrus"
)

// GrantVmAccess adds access to a file for a given VM
func GrantVmAccess(vmid string, filepath string) (err error) {
	title := "hcsshim::GrantVmAccess"
	fields := logrus.Fields{
		"vm-id": vmid,
		"path":  filepath,
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

	err = grantVmAccess(vmid, filepath)
	if err != nil {
		return hcserror.New(err, title+" - failed", "")
	}
	return nil
}
