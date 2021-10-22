package applicationcredentials

import "github.com/gophercloud/gophercloud"

func listURL(client *gophercloud.ServiceClient, userID string) string {
	return client.ServiceURL("users", userID, "application_credentials")
}

func getURL(client *gophercloud.ServiceClient, userID string, id string) string {
	return client.ServiceURL("users", userID, "application_credentials", id)
}

func createURL(client *gophercloud.ServiceClient, userID string) string {
	return client.ServiceURL("users", userID, "application_credentials")
}

func deleteURL(client *gophercloud.ServiceClient, userID string, id string) string {
	return client.ServiceURL("users", userID, "application_credentials", id)
}
