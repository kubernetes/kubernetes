package volumeattach

import (
	"errors"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// List returns a Pager that allows you to iterate over a collection of VolumeAttachments.
func List(client *gophercloud.ServiceClient, serverId string) pagination.Pager {
	return pagination.NewPager(client, listURL(client, serverId), func(r pagination.PageResult) pagination.Page {
		return VolumeAttachmentsPage{pagination.SinglePageBase(r)}
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
	Device string

	// VolumeID is the ID of the volume to attach to the instance
	VolumeID string
}

// ToVolumeAttachmentCreateMap constructs a request body from CreateOpts.
func (opts CreateOpts) ToVolumeAttachmentCreateMap() (map[string]interface{}, error) {
	if opts.VolumeID == "" {
		return nil, errors.New("Missing field required for volume attachment creation: VolumeID")
	}

	volumeAttachment := make(map[string]interface{})
	volumeAttachment["volumeId"] = opts.VolumeID
	if opts.Device != "" {
		volumeAttachment["device"] = opts.Device
	}

	return map[string]interface{}{"volumeAttachment": volumeAttachment}, nil
}

// Create requests the creation of a new volume attachment on the server
func Create(client *gophercloud.ServiceClient, serverId string, opts CreateOptsBuilder) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToVolumeAttachmentCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = client.Post(createURL(client, serverId), reqBody, &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return res
}

// Get returns public data about a previously created VolumeAttachment.
func Get(client *gophercloud.ServiceClient, serverId, aId string) GetResult {
	var res GetResult
	_, res.Err = client.Get(getURL(client, serverId, aId), &res.Body, nil)
	return res
}

// Delete requests the deletion of a previous stored VolumeAttachment from the server.
func Delete(client *gophercloud.ServiceClient, serverId, aId string) DeleteResult {
	var res DeleteResult
	_, res.Err = client.Delete(deleteURL(client, serverId, aId), nil)
	return res
}
