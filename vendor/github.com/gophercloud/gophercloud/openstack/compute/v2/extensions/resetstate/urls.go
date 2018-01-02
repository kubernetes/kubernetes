package resetstate

import (
	"github.com/gophercloud/gophercloud"
)

func actionURL(client *gophercloud.ServiceClient, id string) string {
	return client.ServiceURL("servers", id, "action")
}
