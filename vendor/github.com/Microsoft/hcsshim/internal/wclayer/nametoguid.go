package wclayer

import (
	"github.com/Microsoft/go-winio/pkg/guid"
	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/sirupsen/logrus"
)

// NameToGuid converts the given string into a GUID using the algorithm in the
// Host Compute Service, ensuring GUIDs generated with the same string are common
// across all clients.
func NameToGuid(name string) (id guid.GUID, err error) {
	title := "hcsshim::NameToGuid"
	fields := logrus.Fields{
		"name": name,
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

	err = nameToGuid(name, &id)
	if err != nil {
		err = hcserror.New(err, title+" - failed", "")
		return
	}
	fields["guid"] = id.String()
	return
}
