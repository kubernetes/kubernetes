package volumes

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"

	"github.com/mitchellh/mapstructure"
)

// Volume contains all the information associated with an OpenStack Volume.
type Volume struct {
	// Current status of the volume.
	Status string `mapstructure:"status"`

	// Human-readable display name for the volume.
	Name string `mapstructure:"display_name"`

	// Instances onto which the volume is attached.
	Attachments []map[string]interface{} `mapstructure:"attachments"`

	// This parameter is no longer used.
	AvailabilityZone string `mapstructure:"availability_zone"`

	// Indicates whether this is a bootable volume.
	Bootable string `mapstructure:"bootable"`

	// The date when this volume was created.
	CreatedAt string `mapstructure:"created_at"`

	// Human-readable description for the volume.
	Description string `mapstructure:"display_description"`

	// The type of volume to create, either SATA or SSD.
	VolumeType string `mapstructure:"volume_type"`

	// The ID of the snapshot from which the volume was created
	SnapshotID string `mapstructure:"snapshot_id"`

	// The ID of another block storage volume from which the current volume was created
	SourceVolID string `mapstructure:"source_volid"`

	// Arbitrary key-value pairs defined by the user.
	Metadata map[string]string `mapstructure:"metadata"`

	// Unique identifier for the volume.
	ID string `mapstructure:"id"`

	// Size of the volume in GB.
	Size int `mapstructure:"size"`
}

// CreateResult contains the response body and error from a Create request.
type CreateResult struct {
	commonResult
}

// GetResult contains the response body and error from a Get request.
type GetResult struct {
	commonResult
}

// DeleteResult contains the response body and error from a Delete request.
type DeleteResult struct {
	gophercloud.ErrResult
}

// ListResult is a pagination.pager that is returned from a call to the List function.
type ListResult struct {
	pagination.SinglePageBase
}

// IsEmpty returns true if a ListResult contains no Volumes.
func (r ListResult) IsEmpty() (bool, error) {
	volumes, err := ExtractVolumes(r)
	if err != nil {
		return true, err
	}
	return len(volumes) == 0, nil
}

// ExtractVolumes extracts and returns Volumes. It is used while iterating over a volumes.List call.
func ExtractVolumes(page pagination.Page) ([]Volume, error) {
	var response struct {
		Volumes []Volume `json:"volumes"`
	}

	err := mapstructure.Decode(page.(ListResult).Body, &response)
	return response.Volumes, err
}

// UpdateResult contains the response body and error from an Update request.
type UpdateResult struct {
	commonResult
}

type commonResult struct {
	gophercloud.Result
}

// Extract will get the Volume object out of the commonResult object.
func (r commonResult) Extract() (*Volume, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		Volume *Volume `json:"volume"`
	}

	err := mapstructure.Decode(r.Body, &res)

	return res.Volume, err
}
