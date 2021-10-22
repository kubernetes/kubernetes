package nodes

import "github.com/gophercloud/gophercloud"

func createURL(client *gophercloud.ServiceClient) string {
	return client.ServiceURL("nodes")
}

func listURL(client *gophercloud.ServiceClient) string {
	return createURL(client)
}

func listDetailURL(client *gophercloud.ServiceClient) string {
	return client.ServiceURL("nodes", "detail")
}

func deleteURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL("nodes", id)
}

func getURL(client *gophercloud.ServiceClient, id string) string {
	return deleteURL(client, id)
}

func updateURL(client *gophercloud.ServiceClient, id string) string {
	return deleteURL(client, id)
}

func validateURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL("nodes", id, "validate")
}

func injectNMIURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL("nodes", id, "management", "inject_nmi")
}

func bootDeviceURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL("nodes", id, "management", "boot_device")
}

func supportedBootDeviceURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL("nodes", id, "management", "boot_device", "supported")
}

func statesResourceURL(client *gophercloud.ServiceClient, id string, state string) string {
	return client.ServiceURL("nodes", id, "states", state)
}

func powerStateURL(client *gophercloud.ServiceClient, id string) string {
	return statesResourceURL(client, id, "power")
}

func provisionStateURL(client *gophercloud.ServiceClient, id string) string {
	return statesResourceURL(client, id, "provision")
}

func raidConfigURL(client *gophercloud.ServiceClient, id string) string {
	return statesResourceURL(client, id, "raid")
}
