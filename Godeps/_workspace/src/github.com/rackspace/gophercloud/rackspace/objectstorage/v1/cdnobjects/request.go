package cdnobjects

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/rackspace/objectstorage/v1/cdncontainers"
)

// CDNURL returns the unique CDN URI for the given container and object.
func CDNURL(c *gophercloud.ServiceClient, containerName, objectName string) (string, error) {
	h, err := cdncontainers.Get(c, containerName).Extract()
	if err != nil {
		return "", err
	}
	return h.CDNUri + "/" + objectName, nil
}
