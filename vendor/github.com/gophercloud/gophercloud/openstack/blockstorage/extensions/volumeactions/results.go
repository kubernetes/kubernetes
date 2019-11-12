package volumeactions

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
)

// AttachResult contains the response body and error from an Attach request.
type AttachResult struct {
	gophercloud.ErrResult
}

// BeginDetachingResult contains the response body and error from a BeginDetach
// request.
type BeginDetachingResult struct {
	gophercloud.ErrResult
}

// DetachResult contains the response body and error from a Detach request.
type DetachResult struct {
	gophercloud.ErrResult
}

// UploadImageResult contains the response body and error from an UploadImage
// request.
type UploadImageResult struct {
	gophercloud.Result
}

// SetImageMetadataResult contains the response body and error from an SetImageMetadata
// request.
type SetImageMetadataResult struct {
	gophercloud.ErrResult
}

// ReserveResult contains the response body and error from a Reserve request.
type ReserveResult struct {
	gophercloud.ErrResult
}

// UnreserveResult contains the response body and error from an Unreserve
// request.
type UnreserveResult struct {
	gophercloud.ErrResult
}

// TerminateConnectionResult contains the response body and error from a
// TerminateConnection request.
type TerminateConnectionResult struct {
	gophercloud.ErrResult
}

// InitializeConnectionResult contains the response body and error from an
// InitializeConnection request.
type InitializeConnectionResult struct {
	gophercloud.Result
}

// ExtendSizeResult contains the response body and error from an ExtendSize request.
type ExtendSizeResult struct {
	gophercloud.ErrResult
}

// Extract will get the connection information out of the
// InitializeConnectionResult object.
//
// This will be a generic map[string]interface{} and the results will be
// dependent on the type of connection made.
func (r InitializeConnectionResult) Extract() (map[string]interface{}, error) {
	var s struct {
		ConnectionInfo map[string]interface{} `json:"connection_info"`
	}
	err := r.ExtractInto(&s)
	return s.ConnectionInfo, err
}

// ImageVolumeType contains volume type information obtained from UploadImage
// action.
type ImageVolumeType struct {
	// The ID of a volume type.
	ID string `json:"id"`

	// Human-readable display name for the volume type.
	Name string `json:"name"`

	// Human-readable description for the volume type.
	Description string `json:"display_description"`

	// Flag for public access.
	IsPublic bool `json:"is_public"`

	// Extra specifications for volume type.
	ExtraSpecs map[string]interface{} `json:"extra_specs"`

	// ID of quality of service specs.
	QosSpecsID string `json:"qos_specs_id"`

	// Flag for deletion status of volume type.
	Deleted bool `json:"deleted"`

	// The date when volume type was deleted.
	DeletedAt time.Time `json:"-"`

	// The date when volume type was created.
	CreatedAt time.Time `json:"-"`

	// The date when this volume was last updated.
	UpdatedAt time.Time `json:"-"`
}

func (r *ImageVolumeType) UnmarshalJSON(b []byte) error {
	type tmp ImageVolumeType
	var s struct {
		tmp
		CreatedAt gophercloud.JSONRFC3339MilliNoZ `json:"created_at"`
		UpdatedAt gophercloud.JSONRFC3339MilliNoZ `json:"updated_at"`
		DeletedAt gophercloud.JSONRFC3339MilliNoZ `json:"deleted_at"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = ImageVolumeType(s.tmp)

	r.CreatedAt = time.Time(s.CreatedAt)
	r.UpdatedAt = time.Time(s.UpdatedAt)
	r.DeletedAt = time.Time(s.DeletedAt)

	return err
}

// VolumeImage contains information about volume uploaded to an image service.
type VolumeImage struct {
	// The ID of a volume an image is created from.
	VolumeID string `json:"id"`

	// Container format, may be bare, ofv, ova, etc.
	ContainerFormat string `json:"container_format"`

	// Disk format, may be raw, qcow2, vhd, vdi, vmdk, etc.
	DiskFormat string `json:"disk_format"`

	// Human-readable description for the volume.
	Description string `json:"display_description"`

	// The ID of the created image.
	ImageID string `json:"image_id"`

	// Human-readable display name for the image.
	ImageName string `json:"image_name"`

	// Size of the volume in GB.
	Size int `json:"size"`

	// Current status of the volume.
	Status string `json:"status"`

	// The date when this volume was last updated.
	UpdatedAt time.Time `json:"-"`

	// Volume type object of used volume.
	VolumeType ImageVolumeType `json:"volume_type"`
}

func (r *VolumeImage) UnmarshalJSON(b []byte) error {
	type tmp VolumeImage
	var s struct {
		tmp
		UpdatedAt gophercloud.JSONRFC3339MilliNoZ `json:"updated_at"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = VolumeImage(s.tmp)

	r.UpdatedAt = time.Time(s.UpdatedAt)

	return err
}

// Extract will get an object with info about the uploaded image out of the
// UploadImageResult object.
func (r UploadImageResult) Extract() (VolumeImage, error) {
	var s struct {
		VolumeImage VolumeImage `json:"os-volume_upload_image"`
	}
	err := r.ExtractInto(&s)
	return s.VolumeImage, err
}

// ForceDeleteResult contains the response body and error from a ForceDelete request.
type ForceDeleteResult struct {
	gophercloud.ErrResult
}
