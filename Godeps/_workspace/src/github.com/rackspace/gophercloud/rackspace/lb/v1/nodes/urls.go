package nodes

import (
	"strconv"

	"github.com/rackspace/gophercloud"
)

const (
	lbPath    = "loadbalancers"
	nodePath  = "nodes"
	eventPath = "events"
)

func resourceURL(c *gophercloud.ServiceClient, lbID, nodeID int) string {
	return c.ServiceURL(lbPath, strconv.Itoa(lbID), nodePath, strconv.Itoa(nodeID))
}

func rootURL(c *gophercloud.ServiceClient, lbID int) string {
	return c.ServiceURL(lbPath, strconv.Itoa(lbID), nodePath)
}

func eventsURL(c *gophercloud.ServiceClient, lbID int) string {
	return c.ServiceURL(lbPath, strconv.Itoa(lbID), nodePath, eventPath)
}
