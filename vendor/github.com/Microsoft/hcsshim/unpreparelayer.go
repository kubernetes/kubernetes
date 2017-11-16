package hcsshim

import "github.com/sirupsen/logrus"

// UnprepareLayer disables the filesystem filter for the read-write layer with
// the given id.
func UnprepareLayer(info DriverInfo, layerId string) error {
	title := "hcsshim::UnprepareLayer "
	logrus.Debugf(title+"flavour %d layerId %s", info.Flavour, layerId)

	// Convert info to API calling convention
	infop, err := convertDriverInfo(info)
	if err != nil {
		logrus.Error(err)
		return err
	}

	err = unprepareLayer(&infop, layerId)
	if err != nil {
		err = makeErrorf(err, title, "layerId=%s flavour=%d", layerId, info.Flavour)
		logrus.Error(err)
		return err
	}

	logrus.Debugf(title+"succeeded flavour %d layerId=%s", info.Flavour, layerId)
	return nil
}
