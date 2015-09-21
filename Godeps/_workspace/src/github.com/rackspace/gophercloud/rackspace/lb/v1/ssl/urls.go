package ssl

import (
	"strconv"

	"github.com/rackspace/gophercloud"
)

const (
	path     = "loadbalancers"
	sslPath  = "ssltermination"
	certPath = "certificatemappings"
)

func rootURL(c *gophercloud.ServiceClient, id int) string {
	return c.ServiceURL(path, strconv.Itoa(id), sslPath)
}

func certURL(c *gophercloud.ServiceClient, id int) string {
	return c.ServiceURL(path, strconv.Itoa(id), sslPath, certPath)
}

func certResourceURL(c *gophercloud.ServiceClient, id, certID int) string {
	return c.ServiceURL(path, strconv.Itoa(id), sslPath, certPath, strconv.Itoa(certID))
}
