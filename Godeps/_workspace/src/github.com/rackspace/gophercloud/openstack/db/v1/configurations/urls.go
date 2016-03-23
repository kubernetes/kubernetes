package configurations

import "github.com/rackspace/gophercloud"

func baseURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL("configurations")
}

func resourceURL(c *gophercloud.ServiceClient, configID string) string {
	return c.ServiceURL("configurations", configID)
}

func instancesURL(c *gophercloud.ServiceClient, configID string) string {
	return c.ServiceURL("configurations", configID, "instances")
}

func listDSParamsURL(c *gophercloud.ServiceClient, datastoreID, versionID string) string {
	return c.ServiceURL("datastores", datastoreID, "versions", versionID, "parameters")
}

func getDSParamURL(c *gophercloud.ServiceClient, datastoreID, versionID, paramID string) string {
	return c.ServiceURL("datastores", datastoreID, "versions", versionID, "parameters", paramID)
}

func listGlobalParamsURL(c *gophercloud.ServiceClient, versionID string) string {
	return c.ServiceURL("datastores", "versions", versionID, "parameters")
}

func getGlobalParamURL(c *gophercloud.ServiceClient, versionID, paramID string) string {
	return c.ServiceURL("datastores", "versions", versionID, "parameters", paramID)
}
