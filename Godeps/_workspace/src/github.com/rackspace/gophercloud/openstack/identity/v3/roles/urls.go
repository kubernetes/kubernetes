package roles

import "github.com/rackspace/gophercloud"

func listAssignmentsURL(client *gophercloud.ServiceClient) string {
	return client.ServiceURL("role_assignments")
}
