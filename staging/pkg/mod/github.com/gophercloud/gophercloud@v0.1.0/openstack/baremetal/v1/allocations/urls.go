package allocations

import "github.com/gophercloud/gophercloud"

func createURL(client *gophercloud.ServiceClient) string {
	return client.ServiceURL("allocations")
}

func listURL(client *gophercloud.ServiceClient) string {
	return createURL(client)
}

func resourceURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL("allocations", id)
}

func deleteURL(client *gophercloud.ServiceClient, id string) string {
	return resourceURL(client, id)
}

func getURL(client *gophercloud.ServiceClient, id string) string {
	return resourceURL(client, id)
}
