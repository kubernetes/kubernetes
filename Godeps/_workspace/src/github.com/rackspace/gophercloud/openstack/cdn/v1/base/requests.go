package base

import "github.com/rackspace/gophercloud"

// Get retrieves the home document, allowing the user to discover the
// entire API.
func Get(c *gophercloud.ServiceClient) GetResult {
	var res GetResult
	_, res.Err = c.Get(getURL(c), &res.Body, nil)
	return res
}

// Ping retrieves a ping to the server.
func Ping(c *gophercloud.ServiceClient) PingResult {
	var res PingResult
	_, res.Err = c.Get(pingURL(c), nil, &gophercloud.RequestOpts{
		OkCodes:     []int{204},
		MoreHeaders: map[string]string{"Accept": ""},
	})
	return res
}
