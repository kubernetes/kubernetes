package ports

import "github.com/gophercloud/gophercloud"

func createURL(client *gophercloud.ServiceClient) string {
	return client.ServiceURL("ports")
}

func listURL(client *gophercloud.ServiceClient) string {
	return createURL(client)
}

func listDetailURL(client *gophercloud.ServiceClient) string {
	return client.ServiceURL("ports", "detail")
}

func resourceURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL("ports", id)
}

func deleteURL(client *gophercloud.ServiceClient, id string) string {
	return resourceURL(client, id)
}

func getURL(client *gophercloud.ServiceClient, id string) string {
	return resourceURL(client, id)
}

func updateURL(client *gophercloud.ServiceClient, id string) string {
	return resourceURL(client, id)
}
