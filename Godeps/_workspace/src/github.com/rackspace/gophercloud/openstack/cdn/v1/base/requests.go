package base

import "github.com/rackspace/gophercloud"

// Get retrieves the home document, allowing the user to discover the
// entire API.
func Get(c *gophercloud.ServiceClient) GetResult {
	var res GetResult
	_, res.Err = c.Request("GET", getURL(c), gophercloud.RequestOpts{
		JSONResponse: &res.Body,
		OkCodes:      []int{200},
	})
	return res
}

// Ping retrieves a ping to the server.
func Ping(c *gophercloud.ServiceClient) PingResult {
	var res PingResult
	_, res.Err = c.Request("GET", pingURL(c), gophercloud.RequestOpts{
		OkCodes:     []int{204},
		MoreHeaders: map[string]string{"Accept": ""},
	})
	return res
}
