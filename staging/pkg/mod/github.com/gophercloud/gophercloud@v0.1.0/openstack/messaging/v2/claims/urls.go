package claims

import "github.com/gophercloud/gophercloud"

const (
	apiVersion = "v2"
	apiName    = "queues"
)

func createURL(client *gophercloud.ServiceClient, queueName string) string {
	return client.ServiceURL(apiVersion, apiName, queueName, "claims")
}

func getURL(client *gophercloud.ServiceClient, queueName string, claimID string) string {
	return client.ServiceURL(apiVersion, apiName, queueName, "claims", claimID)
}

func updateURL(client *gophercloud.ServiceClient, queueName string, claimID string) string {
	return client.ServiceURL(apiVersion, apiName, queueName, "claims", claimID)
}

func deleteURL(client *gophercloud.ServiceClient, queueName string, claimID string) string {
	return client.ServiceURL(apiVersion, apiName, queueName, "claims", claimID)
}
