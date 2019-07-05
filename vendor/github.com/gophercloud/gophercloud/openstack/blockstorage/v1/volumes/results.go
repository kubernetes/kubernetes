package volumes

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Volume contains all the information associated with an OpenStack Volume.
type Volume struct {
	// Current status of the volume.
	Status string `json:"status"`
	// Human-readable display name for the volume.
	Name string `json:"display_name"`
	// Instances onto which the volume is attached.
	Attachments []map[string]interface{} `json:"attachments"`
	// This parameter is no longer used.
	AvailabilityZone string `json:"availability_zone"`
	// Indicates whether this is a bootable volume.
	Bootable string `json:"bootable"`
	// The date when this volume was created.
	CreatedAt time.Time `json:"-"`
	// Human-readable description for the volume.
	Description string `json:"display_description"`
	// The type of volume to create, either SATA or SSD.
	VolumeType string `json:"volume_type"`
	// The ID of the snapshot from which the volume was created
	SnapshotID string `json:"snapshot_id"`
	// The ID of another block storage volume from which the current volume was created
	SourceVolID string `json:"source_volid"`
	// Arbitrary key-value pairs defined by the user.
	Metadata map[string]string `json:"metadata"`
	// Unique identifier for the volume.
	ID string `json:"id"`
	// Size of the volume in GB.
	Size int `json:"size"`
}

func (r *Volume) UnmarshalJSON(b []byte) error {
	type tmp Volume
	var s struct {
		tmp
		CreatedAt gophercloud.JSONRFC3339MilliNoZ `json:"created_at"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = Volume(s.tmp)

	r.CreatedAt = time.Time(s.CreatedAt)

	return err
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

// VolumePage is a pagination.pager that is returned from a call to the List function.
type VolumePage struct {
	pagination.SinglePageBase
}

// IsEmpty returns true if a VolumePage contains no Volumes.
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

// UpdateResult contains the response body and error from an Update request.
type UpdateResult struct {
	commonResult
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
