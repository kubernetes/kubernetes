package clusters

import "github.com/gophercloud/gophercloud"

var apiVersion = "v1"
var apiName = "clusters"

func commonURL(client *gophercloud.ServiceClient) string {
	return client.ServiceURL(apiVersion, apiName)
}

func idURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL(apiVersion, apiName, id)
}

func actionURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL(apiVersion, apiName, id, "actions")
}

func createURL(client *gophercloud.ServiceClient) string {
	return commonURL(client)
}

func getURL(client *gophercloud.ServiceClient, id string) string {
	return idURL(client, id)
}

func listURL(client *gophercloud.ServiceClient) string {
	return commonURL(client)
}

func updateURL(client *gophercloud.ServiceClient, id string) string {
	return idURL(client, id)
}

func deleteURL(client *gophercloud.ServiceClient, id string) string {
	return idURL(client, id)
}

func listPoliciesURL(client *gophercloud.ServiceClient, clusterID string) string {
	return client.ServiceURL(apiVersion, apiName, clusterID, "policies")
}

func getPolicyURL(client *gophercloud.ServiceClient, clusterID string, policyID string) string {
	return client.ServiceURL(apiVersion, apiName, clusterID, "policies", policyID)
}

func nodeURL(client *gophercloud.ServiceClient, id string) string {
	return actionURL(client, id)
}

func collectURL(client *gophercloud.ServiceClient, clusterID string, path string) string {
	return client.ServiceURL(apiVersion, apiName, clusterID, "attrs", path)
}

func opsURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL(apiVersion, apiName, id, "ops")
}
