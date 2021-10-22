package profiletypes

import "github.com/gophercloud/gophercloud"

const (
	apiVersion = "v1"
	apiName    = "profile-types"
)

func commonURL(client *gophercloud.ServiceClient) string {
	return client.ServiceURL(apiVersion, apiName)
}

func profileTypeURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL(apiVersion, apiName, id)
}

func getURL(client *gophercloud.ServiceClient, id string) string {
	return profileTypeURL(client, id)
}

func listURL(client *gophercloud.ServiceClient) string {
	return commonURL(client)
}

func listOpsURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL(apiVersion, apiName, id, "ops")
}
