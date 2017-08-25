package hcsshim

import "github.com/sirupsen/logrus"

// CreateSandboxLayer creates and populates new read-write layer for use by a container.
// This requires both the id of the direct parent layer, as well as the full list
// of paths to all parent layers up to the base (and including the direct parent
// whose id was provided).
func CreateSandboxLayer(info DriverInfo, layerId, parentId string, parentLayerPaths []string) error {
	title := "hcsshim::CreateSandboxLayer "
	logrus.Debugf(title+"layerId %s parentId %s", layerId, parentId)

	// Generate layer descriptors
	layers, err := layerPathsToDescriptors(parentLayerPaths)
	if err != nil {
		return err
	}

	// Convert info to API calling convention
	infop, err := convertDriverInfo(info)
	if err != nil {
		logrus.Error(err)
		return err
	}

	err = createSandboxLayer(&infop, layerId, parentId, layers)
	if err != nil {
		err = makeErrorf(err, title, "layerId=%s parentId=%s", layerId, parentId)
		logrus.Error(err)
		return err
	}

	logrus.Debugf(title+"- succeeded layerId=%s parentId=%s", layerId, parentId)
	return nil
}
