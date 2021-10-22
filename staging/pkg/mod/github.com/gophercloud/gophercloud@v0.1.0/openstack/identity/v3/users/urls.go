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

func changePasswordURL(client *gophercloud.ServiceClient, userID string) string {
	return client.ServiceURL("users", userID, "password")
}

func deleteURL(client *gophercloud.ServiceClient, userID string) string {
	return client.ServiceURL("users", userID)
}

func listGroupsURL(client *gophercloud.ServiceClient, userID string) string {
	return client.ServiceURL("users", userID, "groups")
}

func addToGroupURL(client *gophercloud.ServiceClient, groupID, userID string) string {
	return client.ServiceURL("groups", groupID, "users", userID)
}

func isMemberOfGroupURL(client *gophercloud.ServiceClient, groupID, userID string) string {
	return client.ServiceURL("groups", groupID, "users", userID)
}

func removeFromGroupURL(client *gophercloud.ServiceClient, groupID, userID string) string {
	return client.ServiceURL("groups", groupID, "users", userID)
}

func listProjectsURL(client *gophercloud.ServiceClient, userID string) string {
	return client.ServiceURL("users", userID, "projects")
}

func listInGroupURL(client *gophercloud.ServiceClient, groupID string) string {
	return client.ServiceURL("groups", groupID, "users")
}
