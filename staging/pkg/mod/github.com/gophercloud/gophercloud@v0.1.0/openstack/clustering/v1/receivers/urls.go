package receivers

import "github.com/gophercloud/gophercloud"

var apiVersion = "v1"
var apiName = "receivers"

func commonURL(client *gophercloud.ServiceClient) string {
	return client.ServiceURL(apiVersion, apiName)
}

func createURL(client *gophercloud.ServiceClient) string {
	return commonURL(client)
}

func idURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL(apiVersion, apiName, id)
}

func getURL(client *gophercloud.ServiceClient, id string) string {
	return idURL(client, id)
}

func updateURL(client *gophercloud.ServiceClient, id string) string {
	return idURL(client, id)
}

func listURL(client *gophercloud.ServiceClient) string {
	return commonURL(client)
}

func deleteURL(client *gophercloud.ServiceClient, id string) string {
	return idURL(client, id)
}

func notifyURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL(apiVersion, apiName, id, "notify")
}
