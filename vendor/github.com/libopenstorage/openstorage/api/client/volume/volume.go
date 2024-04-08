package volume

import (
	"crypto/tls"
	"fmt"

	"github.com/libopenstorage/openstorage/api"
	"github.com/libopenstorage/openstorage/api/client"
	"github.com/libopenstorage/openstorage/volume"
)

// VolumeDriver returns a REST wrapper for the VolumeDriver interface.
func VolumeDriver(c *client.Client) volume.VolumeDriver {
	return newVolumeClient(c)
}

// NewAuthDriverClient returns a new REST client of the supplied version for specified driver.
// host: REST endpoint [http://<ip>:<port> OR unix://<path-to-unix-socket>]. default: [unix:///var/lib/osd/<driverName>.sock]
// version: Volume API version
// authstring can be set to the JWT Token and accesstoken set to an empty string.
func NewAuthDriverClient(host, driverName, version, authstring, accesstoken, userAgent string) (*client.Client, error) {
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
	return client.NewAuthClient(host, version, authstring, accesstoken, userAgent)
}

// NewDriverClient returns a new REST client of the supplied version for specified driver.
// host: REST endpoint [http://<ip>:<port> OR unix://<path-to-unix-socket>]. default: [unix:///var/lib/osd/<driverName>.sock]
// version: Volume API version
// userAgent: Drivername for http connections
func NewDriverClient(host, driverName, version, userAgent string) (*client.Client, error) {
	if host == "" {
		if driverName == "" {
			return nil, fmt.Errorf("Driver Name cannot be empty")
		}
		host = client.GetUnixServerPath(driverName, volume.DriverAPIBase)
	}
	if version == "" {
		// Set the default version
		version = volume.APIVersion
	}
	return client.NewClient(host, version, userAgent)
}

// GetAuthSupportedDriverVersions returns a list of supported versions
// for the provided driver. It uses the given security params and
// server endpoint or the standard unix domain socket
// authstring can be set to the JWT Token and accesstoken set to an empty string.
func GetAuthSupportedDriverVersions(driverName, host, authstring, accesstoken string, tlsConfig *tls.Config) ([]string, error) {
	// Get a client handler
	if host == "" {
		host = client.GetUnixServerPath(driverName, volume.DriverAPIBase)
	}

	client, err := client.NewAuthClient(host, "", authstring, accesstoken, "")
	if err != nil {
		return []string{}, err
	}

	if tlsConfig != nil {
		client.SetTLS(tlsConfig)
	}

	versions, err := client.Versions(api.OsdVolumePath)
	if err != nil {
		return []string{}, err
	}
	return versions, nil
}

// GetSupportedDriverVersions returns a list of supported versions
// for the provided driver. It uses the given server endpoint or the
// standard unix domain socket
func GetSupportedDriverVersions(driverName, host string) ([]string, error) {
	// Get a client handler
	if host == "" {
		host = client.GetUnixServerPath(driverName, volume.DriverAPIBase)
	}

	client, err := client.NewClient(host, "", "")
	if err != nil {
		return []string{}, err
	}
	versions, err := client.Versions(api.OsdVolumePath)
	if err != nil {
		return []string{}, err
	}
	return versions, nil
}
