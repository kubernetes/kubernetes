package hcsshim

import "github.com/sirupsen/logrus"

// ActivateLayer will find the layer with the given id and mount it's filesystem.
// For a read/write layer, the mounted filesystem will appear as a volume on the
// host, while a read-only layer is generally expected to be a no-op.
// An activated layer must later be deactivated via DeactivateLayer.
func ActivateLayer(info DriverInfo, id string) error {
	title := "hcsshim::ActivateLayer "
	logrus.Debugf(title+"Flavour %d ID %s", info.Flavour, id)

	infop, err := convertDriverInfo(info)
	if err != nil {
		logrus.Error(err)
		return err
	}

	err = activateLayer(&infop, id)
	if err != nil {
		err = makeErrorf(err, title, "id=%s flavour=%d", id, info.Flavour)
		logrus.Error(err)
		return err
	}

	logrus.Debugf(title+" - succeeded id=%s flavour=%d", id, info.Flavour)
	return nil
}
