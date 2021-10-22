package sharetypes

import "github.com/gophercloud/gophercloud"

func createURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL("types")
}

func deleteURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("types", id)
}

func listURL(c *gophercloud.ServiceClient) string {
	return createURL(c)
}

func getDefaultURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL("types", "default")
}

func getExtraSpecsURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("types", id, "extra_specs")
}

func setExtraSpecsURL(c *gophercloud.ServiceClient, id string) string {
	return getExtraSpecsURL(c, id)
}

func unsetExtraSpecsURL(c *gophercloud.ServiceClient, id string, key string) string {
	return c.ServiceURL("types", id, "extra_specs", key)
}

func showAccessURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("types", id, "share_type_access")
}

func addAccessURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("types", id, "action")
}

func removeAccessURL(c *gophercloud.ServiceClient, id string) string {
	return addAccessURL(c, id)
}
