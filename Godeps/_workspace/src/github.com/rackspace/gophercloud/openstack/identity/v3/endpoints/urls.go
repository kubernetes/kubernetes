package endpoints

import "github.com/rackspace/gophercloud"

func listURL(client *gophercloud.ServiceClient) string {
	return client.ServiceURL("endpoints")
}

func endpointURL(client *gophercloud.ServiceClient, endpointID string) string {
	return client.ServiceURL("endpoints", endpointID)
}
