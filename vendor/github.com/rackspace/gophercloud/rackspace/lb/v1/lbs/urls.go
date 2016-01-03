package lbs

import (
	"strconv"

	"github.com/rackspace/gophercloud"
)

const (
	path           = "loadbalancers"
	protocolsPath  = "protocols"
	algorithmsPath = "algorithms"
	logPath        = "connectionlogging"
	epPath         = "errorpage"
	stPath         = "stats"
	cachePath      = "contentcaching"
)

func resourceURL(c *gophercloud.ServiceClient, id int) string {
	return c.ServiceURL(path, strconv.Itoa(id))
}

func rootURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL(path)
}

func protocolsURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL(path, protocolsPath)
}

func algorithmsURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL(path, algorithmsPath)
}

func loggingURL(c *gophercloud.ServiceClient, id int) string {
	return c.ServiceURL(path, strconv.Itoa(id), logPath)
}

func errorPageURL(c *gophercloud.ServiceClient, id int) string {
	return c.ServiceURL(path, strconv.Itoa(id), epPath)
}

func statsURL(c *gophercloud.ServiceClient, id int) string {
	return c.ServiceURL(path, strconv.Itoa(id), stPath)
}

func cacheURL(c *gophercloud.ServiceClient, id int) string {
	return c.ServiceURL(path, strconv.Itoa(id), cachePath)
}
