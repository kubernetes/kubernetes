package workflows

import (
	"github.com/gophercloud/gophercloud"
)

func createURL(client *gophercloud.ServiceClient) string {
	return client.ServiceURL("workflows")
}

func deleteURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL("workflows", id)
}

func getURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL("workflows", id)
}

func listURL(client *gophercloud.ServiceClient) string {
	return client.ServiceURL("workflows")
}
