package hcsshim

import "github.com/sirupsen/logrus"

// CreateLayer creates a new, empty, read-only layer on the filesystem based on
// the parent layer provided.
func CreateLayer(info DriverInfo, id, parent string) error {
	title := "hcsshim::CreateLayer "
	logrus.Debugf(title+"Flavour %d ID %s parent %s", info.Flavour, id, parent)

	// Convert info to API calling convention
	infop, err := convertDriverInfo(info)
	if err != nil {
		logrus.Error(err)
		return err
	}

	err = createLayer(&infop, id, parent)
	if err != nil {
		err = makeErrorf(err, title, "id=%s parent=%s flavour=%d", id, parent, info.Flavour)
		logrus.Error(err)
		return err
	}

	logrus.Debugf(title+" - succeeded id=%s parent=%s flavour=%d", id, parent, info.Flavour)
	return nil
}
