package vips

import (
	"strconv"

	"github.com/rackspace/gophercloud"
)

const (
	lbPath  = "loadbalancers"
	vipPath = "virtualips"
)

func resourceURL(c *gophercloud.ServiceClient, lbID, nodeID int) string {
	return c.ServiceURL(lbPath, strconv.Itoa(lbID), vipPath, strconv.Itoa(nodeID))
}

func rootURL(c *gophercloud.ServiceClient, lbID int) string {
	return c.ServiceURL(lbPath, strconv.Itoa(lbID), vipPath)
}
