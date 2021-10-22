package secrets

import "github.com/gophercloud/gophercloud"

func listURL(client *gophercloud.ServiceClient) string {
	return client.ServiceURL("secrets")
}

func getURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL("secrets", id)
}

func createURL(client *gophercloud.ServiceClient) string {
	return client.ServiceURL("secrets")
}

func deleteURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL("secrets", id)
}

func updateURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL("secrets", id)
}

func payloadURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL("secrets", id, "payload")
}

func metadataURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL("secrets", id, "metadata")
}

func metadatumURL(client *gophercloud.ServiceClient, id, key string) string {
	return client.ServiceURL("secrets", id, "metadata", key)
}
