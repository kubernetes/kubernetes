package volume

import (
	"fmt"
	"github.com/libopenstorage/openstorage/api/client"
	"github.com/libopenstorage/openstorage/volume"
	"github.com/libopenstorage/openstorage/api"
)

// VolumeDriver returns a REST wrapper for the VolumeDriver interface.
func VolumeDriver(c *client.Client) volume.VolumeDriver {
	return newVolumeClient(c)
}

// NewDriver returns a new REST client of the supplied version for specified driver.
// host: REST endpoint [http://<ip>:<port> OR unix://<path-to-unix-socket>]. default: [unix:///var/lib/osd/<driverName>.sock]
// version: Volume API version
func NewDriverClient(host, driverName, version string) (*client.Client, error) {
	if driverName == "" {
		return nil, fmt.Errorf("Driver Name cannot be empty")
	}
	if host == "" {
		host = client.GetUnixServerPath(driverName, volume.DriverAPIBase)
	}
	if version == "" {
		// Set the default version
		version = volume.APIVersion
	}
	return client.NewClient(host, version)
}

// GetSupportedDriverVersions returns a list of supported versions
// for the provided driver. It uses the given server endpoint or the
// standard unix domain socket
func GetSupportedDriverVersions(driverName, host string) ([]string, error) {
	// Get a client handler
	if host == "" {
		host = client.GetUnixServerPath(driverName, volume.DriverAPIBase)
	}

	client, err := client.NewClient(host, "")
	if err != nil {
		return []string{}, err
	}
	versions, err := client.Versions(api.OsdVolumePath)
	if err != nil {
		return []string{}, err
	}
	return versions, nil
}
