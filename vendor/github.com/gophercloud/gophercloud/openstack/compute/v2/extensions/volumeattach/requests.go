package volumeattach

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// List returns a Pager that allows you to iterate over a collection of VolumeAttachments.
func List(client *gophercloud.ServiceClient, serverID string) pagination.Pager {
	return pagination.NewPager(client, listURL(client, serverID), func(r pagination.PageResult) pagination.Page {
		return VolumeAttachmentPage{pagination.SinglePageBase(r)}
	})
}

// CreateOptsBuilder describes struct types that can be accepted by the Create call. Notable, the
// CreateOpts struct in this package does.
type CreateOptsBuilder interface {
	ToVolumeAttachmentCreateMap() (map[string]interface{}, error)
}

// CreateOpts specifies volume attachment creation or import parameters.
type CreateOpts struct {
	// Device is the device that the volume will attach to the instance as. Omit for "auto"
	Device string `json:"device,omitempty"`
	// VolumeID is the ID of the volume to attach to the instance
	VolumeID string `json:"volumeId" required:"true"`
}

// ToVolumeAttachmentCreateMap constructs a request body from CreateOpts.
func (opts CreateOpts) ToVolumeAttachmentCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "volumeAttachment")
}

// Create requests the creation of a new volume attachment on the server
func Create(client *gophercloud.ServiceClient, serverID string, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToVolumeAttachmentCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(createURL(client, serverID), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// Get returns public data about a previously created VolumeAttachment.
func Get(client *gophercloud.ServiceClient, serverID, attachmentID string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, serverID, attachmentID), &r.Body, nil)
	return
}

// Delete requests the deletion of a previous stored VolumeAttachment from the server.
func Delete(client *gophercloud.ServiceClient, serverID, attachmentID string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, serverID, attachmentID), nil)
	return
}
