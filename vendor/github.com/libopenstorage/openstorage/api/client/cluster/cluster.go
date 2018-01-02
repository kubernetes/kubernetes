package cluster

import (
	"github.com/libopenstorage/openstorage/api/client"
	"github.com/libopenstorage/openstorage/cluster"
)

const (
	// OsdSocket is the unix socket for cluster apis
	OsdSocket = "osd"
)

// ClusterManager returns a REST wrapper for the Cluster interface.
func ClusterManager(c *client.Client) cluster.Cluster {
	return newClusterClient(c)
}

// NewAuthClusterClient returns a new REST client.
// host: REST endpoint [http://<ip>:<port> OR unix://<path-to-unix-socket>]. default: [unix://var/lib/osd/cluster/osd.sock]
// version: Cluster API version
func NewAuthClusterClient(host, version string, authstring string, accesstoken string) (*client.Client, error) {
	if host == "" {
		host = client.GetUnixServerPath(OsdSocket, cluster.APIBase)
	}

	if version == "" {
		// Set the default version
		version = cluster.APIVersion
	}

	return client.NewAuthClient(host, version, authstring, accesstoken, "")
}

// NewClusterClient returns a new REST client.
// host: REST endpoint [http://<ip>:<port> OR unix://<path-to-unix-socket>]. default: [unix://var/lib/osd/cluster/osd.sock]
// version: Cluster API version
func NewClusterClient(host, version string) (*client.Client, error) {
	if host == "" {
		host = client.GetUnixServerPath(OsdSocket, cluster.APIBase)
	}

	if version == "" {
		// Set the default version
		version = cluster.APIVersion
	}

	return client.NewClient(host, version, "")
}

// GetSupportedClusterVersions returns a list of supported versions of the Cluster API
// host: REST endpoint [http://<ip>:<port> OR unix://<path-to-unix-socket>]. default: [unix://var/lib/osd/cluster/osd.sock]
func GetSupportedClusterVersions(host string) ([]string, error) {
	if host == "" {
		host = client.GetUnixServerPath(OsdSocket, cluster.APIBase)
	}
	client, err := client.NewClient(host, "", "")
	if err != nil {
		return []string{}, err
	}
	versions, err := client.Versions("cluster")
	if err != nil {
		return []string{}, err
	}
	return versions, nil
}
