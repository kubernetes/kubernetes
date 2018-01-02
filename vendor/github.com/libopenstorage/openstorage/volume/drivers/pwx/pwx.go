package pwx

import (
	"github.com/libopenstorage/openstorage/api"
	"github.com/libopenstorage/openstorage/api/client"
	volumeclient "github.com/libopenstorage/openstorage/api/client/volume"
	"github.com/libopenstorage/openstorage/config"
	"github.com/libopenstorage/openstorage/volume"
)

const (
	// Name of the driver
	Name = "pwx"
	// Type of the driver
	Type = api.DriverType_DRIVER_TYPE_BLOCK
	// DefaultUrl where the driver's socket resides
	DefaultUrl = "unix:///" + volume.DriverAPIBase + "pxd.sock"
)

type driver struct {
	volume.VolumeDriver
}

// Init initialized the Portworx driver.
// Portworx natively implements the openstorage.org API specification, so
// we can directly point the VolumeDriver to the PWX API server.
func Init(params map[string]string) (volume.VolumeDriver, error) {
	url, ok := params[config.UrlKey]
	if !ok {
		url = DefaultUrl
	}
	version, ok := params[config.VersionKey]
	if !ok {
		version = volume.APIVersion
	}
	c, err := client.NewClient(url, version, "")
	if err != nil {
		return nil, err
	}

	return &driver{VolumeDriver: volumeclient.VolumeDriver(c)}, nil
}

func (d *driver) Name() string {
	return Name
}

func (d *driver) Type() api.DriverType {
	return Type
}
