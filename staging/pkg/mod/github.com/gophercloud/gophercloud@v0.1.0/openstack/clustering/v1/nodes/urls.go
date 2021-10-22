package nodes

import "github.com/gophercloud/gophercloud"

var apiVersion = "v1"
var apiName = "nodes"

func commonURL(client *gophercloud.ServiceClient) string {
	return client.ServiceURL(apiVersion, apiName)
}

func actionURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL(apiVersion, apiName, id, "actions")
}

func createURL(client *gophercloud.ServiceClient) string {
	return commonURL(client)
}

func listURL(client *gophercloud.ServiceClient) string {
	return commonURL(client)
}

func idURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL(apiVersion, apiName, id)
}

func deleteURL(client *gophercloud.ServiceClient, id string) string {
	return idURL(client, id)
}

func getURL(client *gophercloud.ServiceClient, id string) string {
	return idURL(client, id)
}

func updateURL(client *gophercloud.ServiceClient, id string) string {
	return idURL(client, id)
}

func opsURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL(apiVersion, apiName, id, "ops")
}
