package users

import "github.com/rackspace/gophercloud"

func baseURL(c *gophercloud.ServiceClient, instanceID string) string {
	return c.ServiceURL("instances", instanceID, "users")
}

func userURL(c *gophercloud.ServiceClient, instanceID, userName string) string {
	return c.ServiceURL("instances", instanceID, "users", userName)
}

func dbsURL(c *gophercloud.ServiceClient, instanceID, userName string) string {
	return c.ServiceURL("instances", instanceID, "users", userName, "databases")
}

func dbURL(c *gophercloud.ServiceClient, instanceID, userName, dbName string) string {
	return c.ServiceURL("instances", instanceID, "users", userName, "databases", dbName)
}
