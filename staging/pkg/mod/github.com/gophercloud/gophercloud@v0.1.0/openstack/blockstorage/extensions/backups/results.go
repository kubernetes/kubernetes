package backups

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Backup contains all the information associated with a Cinder Backup.
type Backup struct {
	// ID is the Unique identifier of the backup.
	ID string `json:"id"`

	// CreatedAt is the date the backup was created.
	CreatedAt time.Time `json:"-"`

	// UpdatedAt is the date the backup was updated.
	UpdatedAt time.Time `json:"-"`

	// Name is the display name of the backup.
	Name string `json:"name"`

	// Description is the description of the backup.
	Description string `json:"description"`

	// VolumeID is the ID of the Volume from which this backup was created.
	VolumeID string `json:"volume_id"`

	// SnapshotID is the ID of the snapshot from which this backup was created.
	SnapshotID string `json:"snapshot_id"`

	// Status is the status of the backup.
	Status string `json:"status"`

	// Size is the size of the backup, in GB.
	Size int `json:"size"`

	// Object Count is the number of objects in the backup.
	ObjectCount int `json:"object_count"`

	// Container is the container where the backup is stored.
	Container string `json:"container"`

	// AvailabilityZone is the availability zone of the backup.
	AvailabilityZone string `json:"availability_zone"`

	// HasDependentBackups is whether there are other backups
	// depending on this backup.
	HasDependentBackups bool `json:"has_dependent_backups"`

	// FailReason has the reason for the backup failure.
	FailReason string `json:"fail_reason"`

	// IsIncremental is whether this is an incremental backup.
	IsIncremental bool `json:"is_incremental"`

	// DataTimestamp is the time when the data on the volume was first saved.
	DataTimestamp time.Time `json:"-"`

	// ProjectID is the ID of the project that owns the backup. This is
	// an admin-only field.
	ProjectID string `json:"os-backup-project-attr:project_id"`
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

// BackupPage is a pagination.Pager that is returned from a call to the List function.
type BackupPage struct {
	pagination.LinkedPageBase
}

// UnmarshalJSON converts our JSON API response into our backup struct
func (r *Backup) UnmarshalJSON(b []byte) error {
	type tmp Backup
	var s struct {
		tmp
		CreatedAt     gophercloud.JSONRFC3339MilliNoZ `json:"created_at"`
		UpdatedAt     gophercloud.JSONRFC3339MilliNoZ `json:"updated_at"`
		DataTimestamp gophercloud.JSONRFC3339MilliNoZ `json:"data_timestamp"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = Backup(s.tmp)

	r.CreatedAt = time.Time(s.CreatedAt)
	r.UpdatedAt = time.Time(s.UpdatedAt)
	r.DataTimestamp = time.Time(s.DataTimestamp)

	return err
}

// IsEmpty returns true if a BackupPage contains no Backups.
func (r BackupPage) IsEmpty() (bool, error) {
	volumes, err := ExtractBackups(r)
	return len(volumes) == 0, err
}

func (page BackupPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"backups_links"`
	}
	err := page.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// ExtractBackups extracts and returns Backups. It is used while iterating over a backups.List call.
func ExtractBackups(r pagination.Page) ([]Backup, error) {
	var s []Backup
	err := ExtractBackupsInto(r, &s)
	return s, err
}

// UpdateResult contains the response body and error from an Update request.
type UpdateResult struct {
	commonResult
}

type commonResult struct {
	gophercloud.Result
}

// Extract will get the Backup object out of the commonResult object.
func (r commonResult) Extract() (*Backup, error) {
	var s Backup
	err := r.ExtractInto(&s)
	return &s, err
}

func (r commonResult) ExtractInto(v interface{}) error {
	return r.Result.ExtractIntoStructPtr(v, "backup")
}

func ExtractBackupsInto(r pagination.Page, v interface{}) error {
	return r.(BackupPage).Result.ExtractIntoSlicePtr(v, "backups")
}
