package attachinterfaces

import "github.com/gophercloud/gophercloud"

func listInterfaceURL(client *gophercloud.ServiceClient, serverID string) string {
	return client.ServiceURL("servers", serverID, "os-interface")
}
