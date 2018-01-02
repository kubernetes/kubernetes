package users

import "github.com/gophercloud/gophercloud"

func listURL(client *gophercloud.ServiceClient) string {
	return client.ServiceURL("users")
}

func getURL(client *gophercloud.ServiceClient, userID string) string {
	return client.ServiceURL("users", userID)
}

func createURL(client *gophercloud.ServiceClient) string {
	return client.ServiceURL("users")
}

func updateURL(client *gophercloud.ServiceClient, userID string) string {
	return client.ServiceURL("users", userID)
}

func deleteURL(client *gophercloud.ServiceClient, userID string) string {
	return client.ServiceURL("users", userID)
}

func listGroupsURL(client *gophercloud.ServiceClient, userID string) string {
	return client.ServiceURL("users", userID, "groups")
}

func listProjectsURL(client *gophercloud.ServiceClient, userID string) string {
	return client.ServiceURL("users", userID, "projects")
}

func listInGroupURL(client *gophercloud.ServiceClient, groupID string) string {
	return client.ServiceURL("groups", groupID, "users")
}
