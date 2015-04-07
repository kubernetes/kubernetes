package snapshots

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"

	"github.com/mitchellh/mapstructure"
)

// Snapshot contains all the information associated with an OpenStack Snapshot.
type Snapshot struct {
	// Currect status of the Snapshot.
	Status string `mapstructure:"status"`

	// Display name.
	Name string `mapstructure:"display_name"`

	// Instances onto which the Snapshot is attached.
	Attachments []string `mapstructure:"attachments"`

	// Logical group.
	AvailabilityZone string `mapstructure:"availability_zone"`

	// Is the Snapshot bootable?
	Bootable string `mapstructure:"bootable"`

	// Date created.
	CreatedAt string `mapstructure:"created_at"`

	// Display description.
	Description string `mapstructure:"display_discription"`

	// See VolumeType object for more information.
	VolumeType string `mapstructure:"volume_type"`

	// ID of the Snapshot from which this Snapshot was created.
	SnapshotID string `mapstructure:"snapshot_id"`

	// ID of the Volume from which this Snapshot was created.
	VolumeID string `mapstructure:"volume_id"`

	// User-defined key-value pairs.
	Metadata map[string]string `mapstructure:"metadata"`

	// Unique identifier.
	ID string `mapstructure:"id"`

	// Size of the Snapshot, in GB.
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

// ListResult is a pagination.Pager that is returned from a call to the List function.
type ListResult struct {
	pagination.SinglePageBase
}

// IsEmpty returns true if a ListResult contains no Snapshots.
func (r ListResult) IsEmpty() (bool, error) {
	volumes, err := ExtractSnapshots(r)
	if err != nil {
		return true, err
	}
	return len(volumes) == 0, nil
}

// ExtractSnapshots extracts and returns Snapshots. It is used while iterating over a snapshots.List call.
func ExtractSnapshots(page pagination.Page) ([]Snapshot, error) {
	var response struct {
		Snapshots []Snapshot `json:"snapshots"`
	}

	err := mapstructure.Decode(page.(ListResult).Body, &response)
	return response.Snapshots, err
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
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		Snapshot *Snapshot `json:"snapshot"`
	}

	err := mapstructure.Decode(r.Body, &res)

	return res.Snapshot, err
}
