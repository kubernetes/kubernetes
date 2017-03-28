package servers

import "github.com/gophercloud/gophercloud"

func createURL(client *gophercloud.ServiceClient) string {
	return client.ServiceURL("servers")
}

func listURL(client *gophercloud.ServiceClient) string {
	return createURL(client)
}

func listDetailURL(client *gophercloud.ServiceClient) string {
	return client.ServiceURL("servers", "detail")
}

func deleteURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL("servers", id)
}

func getURL(client *gophercloud.ServiceClient, id string) string {
	return deleteURL(client, id)
}

func updateURL(client *gophercloud.ServiceClient, id string) string {
	return deleteURL(client, id)
}

func actionURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL("servers", id, "action")
}

func metadatumURL(client *gophercloud.ServiceClient, id, key string) string {
	return client.ServiceURL("servers", id, "metadata", key)
}

func metadataURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL("servers", id, "metadata")
}

func listAddressesURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL("servers", id, "ips")
}

func listAddressesByNetworkURL(client *gophercloud.ServiceClient, id, network string) string {
	return client.ServiceURL("servers", id, "ips", network)
}

func passwordURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL("servers", id, "os-server-password")
}
