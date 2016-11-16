package volumes

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"

	"github.com/mitchellh/mapstructure"
)

// Volume contains all the information associated with an OpenStack Volume.
type Volume struct {
	// Instances onto which the volume is attached.
	Attachments []map[string]interface{} `mapstructure:"attachments"`

	// AvailabilityZone is which availability zone the volume is in.
	AvailabilityZone string `mapstructure:"availability_zone"`

	// Indicates whether this is a bootable volume.
	Bootable string `mapstructure:"bootable"`

	// ConsistencyGroupID is the consistency group ID.
	ConsistencyGroupID string `mapstructure:"consistencygroup_id"`

	// The date when this volume was created.
	CreatedAt string `mapstructure:"created_at"`

	// Human-readable description for the volume.
	Description string `mapstructure:"description"`

	// Encrypted denotes if the volume is encrypted.
	Encrypted bool `mapstructure:"encrypted"`

	// Human-readable display name for the volume.
	Name string `mapstructure:"name"`

	// The type of volume to create, either SATA or SSD.
	VolumeType string `mapstructure:"volume_type"`

	// ReplicationDriverData contains data about the replication driver.
	ReplicationDriverData string `mapstructure:"os-volume-replication:driver_data"`

	// ReplicationExtendedStatus contains extended status about replication.
	ReplicationExtendedStatus string `mapstructure:"os-volume-replication:extended_status"`

	// ReplicationStatus is the status of replication.
	ReplicationStatus string `mapstructure:"replication_status"`

	// The ID of the snapshot from which the volume was created
	SnapshotID string `mapstructure:"snapshot_id"`

	// The ID of another block storage volume from which the current volume was created
	SourceVolID string `mapstructure:"source_volid"`

	// Current status of the volume.
	Status string `mapstructure:"status"`

	// TenantID is the id of the project that owns the volume.
	TenantID string `mapstructure:"os-vol-tenant-attr:tenant_id"`

	// Arbitrary key-value pairs defined by the user.
	Metadata map[string]string `mapstructure:"metadata"`

	// Multiattach denotes if the volume is multi-attach capable.
	Multiattach bool `mapstructure:"multiattach"`

	// Unique identifier for the volume.
	ID string `mapstructure:"id"`

	// Size of the volume in GB.
	Size int `mapstructure:"size"`

	// UserID is the id of the user who created the volume.
	UserID string `mapstructure:"user_id"`
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

