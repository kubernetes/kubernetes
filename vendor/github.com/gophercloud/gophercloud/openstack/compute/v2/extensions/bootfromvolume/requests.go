package bootfromvolume

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"
)

type (
	// DestinationType represents the type of medium being used as the
	// destination of the bootable device.
	DestinationType string

	// SourceType represents the type of medium being used as the source of the
	// bootable device.
	SourceType string
)

const (
	// DestinationLocal DestinationType is for using an ephemeral disk as the
	// destination.
	DestinationLocal DestinationType = "local"

	// DestinationVolume DestinationType is for using a volume as the destination.
	DestinationVolume DestinationType = "volume"

	// SourceBlank SourceType is for a "blank" or empty source.
	SourceBlank SourceType = "blank"

	// SourceImage SourceType is for using images as the source of a block device.
	SourceImage SourceType = "image"

	// SourceSnapshot SourceType is for using a volume snapshot as the source of
	// a block device.
	SourceSnapshot SourceType = "snapshot"

	// SourceVolume SourceType is for using a volume as the source of block
	// device.
	SourceVolume SourceType = "volume"
)

// BlockDevice is a structure with options for creating block devices in a
// server. The block device may be created from an image, snapshot, new volume,
// or existing volume. The destination may be a new volume, existing volume
// which will be attached to the instance, ephemeral disk, or boot device.
type BlockDevice struct {
	// SourceType must be one of: "volume", "snapshot", "image", or "blank".
	SourceType SourceType `json:"source_type" required:"true"`

	// UUID is the unique identifier for the existing volume, snapshot, or
	// image (see above).
	UUID string `json:"uuid,omitempty"`

	// BootIndex is the boot index. It defaults to 0.
	BootIndex int `json:"boot_index"`

	// DeleteOnTermination specifies whether or not to delete the attached volume
	// when the server is deleted. Defaults to `false`.
	DeleteOnTermination bool `json:"delete_on_termination"`

	// DestinationType is the type that gets created. Possible values are "volume"
	// and "local".
	DestinationType DestinationType `json:"destination_type,omitempty"`

	// GuestFormat specifies the format of the block device.
	GuestFormat string `json:"guest_format,omitempty"`

	// VolumeSize is the size of the volume to create (in gigabytes). This can be
	// omitted for existing volumes.
	VolumeSize int `json:"volume_size,omitempty"`
}

// CreateOptsExt is a structure that extends the server `CreateOpts` structure
// by allowing for a block device mapping.
type CreateOptsExt struct {
	servers.CreateOptsBuilder
	BlockDevice []BlockDevice `json:"block_device_mapping_v2,omitempty"`
}

// ToServerCreateMap adds the block device mapping option to the base server
// creation options.
func (opts CreateOptsExt) ToServerCreateMap() (map[string]interface{}, error) {
	base, err := opts.CreateOptsBuilder.ToServerCreateMap()
	if err != nil {
		return nil, err
	}

	if len(opts.BlockDevice) == 0 {
		err := gophercloud.ErrMissingInput{}
		err.Argument = "bootfromvolume.CreateOptsExt.BlockDevice"
		return nil, err
	}

	serverMap := base["server"].(map[string]interface{})

	blockDevice := make([]map[string]interface{}, len(opts.BlockDevice))

	for i, bd := range opts.BlockDevice {
		b, err := gophercloud.BuildRequestBody(bd, "")
		if err != nil {
			return nil, err
		}
		blockDevice[i] = b
	}
	serverMap["block_device_mapping_v2"] = blockDevice

	return base, nil
}

// Create requests the creation of a server from the given block device mapping.
func Create(client *gophercloud.ServiceClient, opts servers.CreateOptsBuilder) (r servers.CreateResult) {
	b, err := opts.ToServerCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(createURL(client), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 202},
	})
	return
}
