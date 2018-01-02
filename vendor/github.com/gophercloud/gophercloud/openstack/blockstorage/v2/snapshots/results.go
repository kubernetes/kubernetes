package snapshots

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Snapshot contains all the information associated with a Cinder Snapshot.
type Snapshot struct {
	// Unique identifier.
	ID string `json:"id"`

	// Date created.
	CreatedAt time.Time `json:"-"`

	// Date updated.
	UpdatedAt time.Time `json:"-"`

	// Display name.
	Name string `json:"name"`

	// Display description.
	Description string `json:"description"`

	// ID of the Volume from which this Snapshot was created.
	VolumeID string `json:"volume_id"`

	// Currect status of the Snapshot.
	Status string `json:"status"`

	// Size of the Snapshot, in GB.
	Size int `json:"size"`

	// User-defined key-value pairs.
	Metadata map[string]string `json:"metadata"`
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

// SnapshotPage is a pagination.Pager that is returned from a call to the List function.
type SnapshotPage struct {
	pagination.SinglePageBase
}

func (r *Snapshot) UnmarshalJSON(b []byte) error {
	type tmp Snapshot
	var s struct {
		tmp
		CreatedAt gophercloud.JSONRFC3339MilliNoZ `json:"created_at"`
		UpdatedAt gophercloud.JSONRFC3339MilliNoZ `json:"updated_at"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = Snapshot(s.tmp)

	r.CreatedAt = time.Time(s.CreatedAt)
	r.UpdatedAt = time.Time(s.UpdatedAt)

	return err
}

// IsEmpty returns true if a SnapshotPage contains no Snapshots.
func (r SnapshotPage) IsEmpty() (bool, error) {
	volumes, err := ExtractSnapshots(r)
	return len(volumes) == 0, err
}

// ExtractSnapshots extracts and returns Snapshots. It is used while iterating over a snapshots.List call.
func ExtractSnapshots(r pagination.Page) ([]Snapshot, error) {
	var s struct {
		Snapshots []Snapshot `json:"snapshots"`
	}
	err := (r.(SnapshotPage)).ExtractInto(&s)
	return s.Snapshots, err
}

// UpdateMetadataResult contains the response body and error from an UpdateMetadata request.
type UpdateMetadataResult struct {
	commonResult
}

// ExtractMetadata returns the metadata from a response from snapshots.UpdateMetadata.
func (r UpdateMetadataResult) ExtractMetadata() (map[string]interface{}, error) {
	if r.Err != nil {
		return nil, r.Err
	}
	m := r.Body.(map[string]interface{})["metadata"]
	return m.(map[string]interface{}), nil
}

type commonResult struct {
	gophercloud.Result
}

// Extract will get the Snapshot object out of the commonResult object.
func (r commonResult) Extract() (*Snapshot, error) {
	var s struct {
		Snapshot *Snapshot `json:"snapshot"`
	}
	err := r.ExtractInto(&s)
	return s.Snapshot, err
}
