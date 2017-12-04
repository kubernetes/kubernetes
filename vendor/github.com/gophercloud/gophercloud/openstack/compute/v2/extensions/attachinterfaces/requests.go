package attachinterfaces

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// List makes a request against the nova API to list the server's interfaces.
func List(client *gophercloud.ServiceClient, serverID string) pagination.Pager {
	return pagination.NewPager(client, listInterfaceURL(client, serverID), func(r pagination.PageResult) pagination.Page {
		return InterfacePage{pagination.SinglePageBase(r)}
	})
}

// Get requests details on a single interface attachment by the server and port IDs.
func Get(client *gophercloud.ServiceClient, serverID, portID string) (r GetResult) {
	_, r.Err = client.Get(getInterfaceURL(client, serverID, portID), &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}
