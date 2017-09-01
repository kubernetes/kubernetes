package attachinterfaces

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// List makes a request against the nova API to list the servers interfaces.
func List(client *gophercloud.ServiceClient, serverID string) pagination.Pager {
	return pagination.NewPager(client, listInterfaceURL(client, serverID), func(r pagination.PageResult) pagination.Page {
		return InterfacePage{pagination.SinglePageBase(r)}
	})
}
