package volumeattach

import (
	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/compute/v2/extensions/volumeattach"
	"github.com/rackspace/gophercloud/pagination"
)

// List returns a Pager that allows you to iterate over a collection of VolumeAttachments.
func List(client *gophercloud.ServiceClient, serverID string) pagination.Pager {
	return os.List(client, serverID)
}

// Create requests the creation of a new volume attachment on the server
func Create(client *gophercloud.ServiceClient, serverID string, opts os.CreateOptsBuilder) os.CreateResult {
	return os.Create(client, serverID, opts)
}

// Get returns public data about a previously created VolumeAttachment.
func Get(client *gophercloud.ServiceClient, serverID, aID string) os.GetResult {
	return os.Get(client, serverID, aID)
}

// Delete requests the deletion of a previous stored VolumeAttachment from the server.
func Delete(client *gophercloud.ServiceClient, serverID, aID string) os.DeleteResult {
	return os.Delete(client, serverID, aID)
}
