package hcsshim

import "github.com/sirupsen/logrus"

// DeactivateLayer will dismount a layer that was mounted via ActivateLayer.
func DeactivateLayer(info DriverInfo, id string) error {
	title := "hcsshim::DeactivateLayer "
	logrus.Debugf(title+"Flavour %d ID %s", info.Flavour, id)

	// Convert info to API calling convention
	infop, err := convertDriverInfo(info)
	if err != nil {
		logrus.Error(err)
		return err
	}

	err = deactivateLayer(&infop, id)
	if err != nil {
		err = makeErrorf(err, title, "id=%s flavour=%d", id, info.Flavour)
		logrus.Error(err)
		return err
	}

	logrus.Debugf(title+"succeeded flavour=%d id=%s", info.Flavour, id)
	return nil
}
