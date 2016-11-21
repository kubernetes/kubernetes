package volumes

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

type Attachment struct {
	AttachedAt   gophercloud.JSONRFC3339MilliNoZ `json:"attached_at"`
	AttachmentID string                          `json:"attachment_id"`
	Device       string                          `json:"device"`
	HostName     string                          `json:"host_name"`
	ID           string                          `json:"id"`
	ServerID     string                          `json:"server_id"`
	VolumeID     string                          `json:"volume_id"`
}

// Volume contains all the information associated with an OpenStack Volume.
type Volume struct {
	// Unique identifier for the volume.
	ID string `json:"id"`
	// Current status of the volume.
	Status string `json:"status"`
	// Size of the volume in GB.
	Size int `json:"size"`
	// AvailabilityZone is which availability zone the volume is in.
	AvailabilityZone string `json:"availability_zone"`
	// The date when this volume was created.
	CreatedAt gophercloud.JSONRFC3339MilliNoZ `json:"created_at"`
	// The date when this volume was last updated
	UpdatedAt gophercloud.JSONRFC3339MilliNoZ `json:"updated_at"`
	// Instances onto which the volume is attached.
	Attachments []Attachment `json:"attachments"`
	// Human-readable display name for the volume.
	Name string `json:"name"`
	// Human-readable description for the volume.
	Description string `json:"description"`
	// The type of volume to create, either SATA or SSD.
	VolumeType string `json:"volume_type"`
	// The ID of the snapshot from which the volume was created
	SnapshotID string `json:"snapshot_id"`
	// The ID of another block storage volume from which the current volume was created
	SourceVolID string `json:"source_volid"`
	// Arbitrary key-value pairs defined by the user.
	Metadata map[string]string `json:"metadata"`
	// UserID is the id of the user who created the volume.
	UserID string `json:"user_id"`
	// Indicates whether this is a bootable volume.
	Bootable string `json:"bootable"`
	// Encrypted denotes if the volume is encrypted.
	Encrypted bool `json:"encrypted"`
	// ReplicationStatus is the status of replication.
	ReplicationStatus string `json:"replication_status"`
	// ConsistencyGroupID is the consistency group ID.
	ConsistencyGroupID string `json:"consistencygroup_id"`
	// Multiattach denotes if the volume is multi-attach capable.
	Multiattach bool `json:"multiattach"`
}

/*
THESE BELONG IN EXTENSIONS:
// ReplicationDriverData contains data about the replication driver.
ReplicationDriverData string `json:"os-volume-replication:driver_data"`
// ReplicationExtendedStatus contains extended status about replication.
ReplicationExtendedStatus string `json:"os-volume-replication:extended_status"`
// TenantID is the id of the project that owns the volume.
TenantID string `json:"os-vol-tenant-attr:tenant_id"`
*/

// VolumePage is a pagination.pager that is returned from a call to the List function.
type VolumePage struct {
	pagination.SinglePageBase
}

// IsEmpty returns true if a ListResult contains no Volumes.
func (r VolumePage) IsEmpty() (bool, error) {
	volumes, err := ExtractVolumes(r)
	return len(volumes) == 0, err
}

// ExtractVolumes extracts and returns Volumes. It is used while iterating over a volumes.List call.
func ExtractVolumes(r pagination.Page) ([]Volume, error) {
	var s struct {
		Volumes []Volume `json:"volumes"`
	}
	err := (r.(VolumePage)).ExtractInto(&s)
	return s.Volumes, err
}

type commonResult struct {
	gophercloud.Result
}

// Extract will get the Volume object out of the commonResult object.
func (r commonResult) Extract() (*Volume, error) {
	var s struct {
		Volume *Volume `json:"volume"`
	}
	err := r.ExtractInto(&s)
	return s.Volume, err
}

// CreateResult contains the response body and error from a Create request.
type CreateResult struct {
	commonResult
}

// GetResult contains the response body and error from a Get request.
type GetResult struct {
	commonResult
}

// UpdateResult contains the response body and error from an Update request.
type UpdateResult struct {
	commonResult
}

// DeleteResult contains the response body and error from a Delete request.
type DeleteResult struct {
	gophercloud.ErrResult
}
