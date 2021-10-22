package events

import "github.com/gophercloud/gophercloud"

var apiVersion = "v1"
var apiName = "events"

func commonURL(client *gophercloud.ServiceClient) string {
	return client.ServiceURL(apiVersion, apiName)
}

func listURL(client *gophercloud.ServiceClient) string {
	return commonURL(client)
}

func idURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL(apiVersion, apiName, id)
}

func getURL(client *gophercloud.ServiceClient, id string) string {
	return idURL(client, id)
}
