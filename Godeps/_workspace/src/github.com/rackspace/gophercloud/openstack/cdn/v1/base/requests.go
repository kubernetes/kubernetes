package base

import (
	"github.com/rackspace/gophercloud"

	"github.com/racker/perigee"
)

// Get retrieves the home document, allowing the user to discover the
// entire API.
func Get(c *gophercloud.ServiceClient) GetResult {
	var res GetResult
	_, res.Err = perigee.Request("GET", getURL(c), perigee.Options{
		MoreHeaders: c.AuthenticatedHeaders(),
		Results:     &res.Body,
		OkCodes:     []int{200},
	})
	return res
}

// Ping retrieves a ping to the server.
func Ping(c *gophercloud.ServiceClient) PingResult {
	var res PingResult
	_, res.Err = perigee.Request("GET", pingURL(c), perigee.Options{
		MoreHeaders: c.AuthenticatedHeaders(),
		OkCodes:     []int{204},
		OmitAccept:  true,
	})
	return res
}
