package acl

import (
	"strconv"

	"github.com/rackspace/gophercloud"
)

const (
	path    = "loadbalancers"
	aclPath = "accesslist"
)

func resourceURL(c *gophercloud.ServiceClient, lbID, networkID int) string {
	return c.ServiceURL(path, strconv.Itoa(lbID), aclPath, strconv.Itoa(networkID))
}

func rootURL(c *gophercloud.ServiceClient, lbID int) string {
	return c.ServiceURL(path, strconv.Itoa(lbID), aclPath)
}
