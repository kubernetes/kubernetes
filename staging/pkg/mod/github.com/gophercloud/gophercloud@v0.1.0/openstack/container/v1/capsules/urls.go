package capsules

import "github.com/gophercloud/gophercloud"

func getURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL("capsules", id)
}

func createURL(client *gophercloud.ServiceClient) string {
	return client.ServiceURL("capsules")
}

// `listURL` is a pure function. `listURL(c)` is a URL for which a GET
// request will respond with a list of capsules in the service `c`.
func listURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL("capsules")
}

func deleteURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("capsules", id)
}
