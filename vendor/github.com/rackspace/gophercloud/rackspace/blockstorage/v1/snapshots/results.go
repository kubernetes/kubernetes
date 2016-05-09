package snapshots

import (
	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/blockstorage/v1/snapshots"
	"github.com/rackspace/gophercloud/pagination"

	"github.com/mitchellh/mapstructure"
)

// Status is the type used to represent a snapshot's status
type Status string

// Constants to use for supported statuses
const (
	Creating    Status = "CREATING"
	Available   Status = "AVAILABLE"
	Deleting    Status = "DELETING"
	Error       Status = "ERROR"
	DeleteError Status = "ERROR_DELETING"
)

// Snapshot is the Rackspace representation of an external block storage device.
type Snapshot struct {
	// The timestamp when this snapshot was created.
	CreatedAt string `mapstructure:"created_at"`

	// The human-readable description for this snapshot.
	Description string `mapstructure:"display_description"`

	// The human-readable name for this snapshot.
	Name string `mapstructure:"display_name"`

	// The UUID for this snapshot.
	ID string `mapstructure:"id"`

	// The random metadata associated with this snapshot. Note: unlike standard
	// OpenStack snapshots, this cannot actually be set.
	Metadata map[string]string `mapstructure:"metadata"`

	// Indicates the current progress of the snapshot's backup procedure.
	Progress string `mapstructure:"os-extended-snapshot-attributes:progress"`

	// The project ID.
	ProjectID string `mapstructure:"os-extended-snapshot-attributes:project_id"`

	// The size of the volume which this snapshot backs up.
	Size int `mapstructure:"size"`

	// The status of the snapshot.
	Status Status `mapstructure:"status"`

	// The ID of the volume which this snapshot seeks to back up.
	VolumeID string `mapstructure:"volume_id"`
}

// CreateResult represents the result of a create operation
type CreateResult struct {
	os.CreateResult
}

// GetResult represents the result of a get operation
type GetResult struct {
	os.GetResult
}

// UpdateResult represents the result of an update operation
type UpdateResult struct {
	gophercloud.Result
}

func commonExtract(resp interface{}, err error) (*Snapshot, error) {
	if err != nil {
		return nil, err
	}

	var respStruct struct {
		Snapshot *Snapshot `json:"snapshot"`
	}

	err = mapstructure.Decode(resp, &respStruct)

	return respStruct.Snapshot, err
}

// Extract will get the Snapshot object out of the GetResult object.
func (r GetResult) Extract() (*Snapshot, error) {
	return commonExtract(r.Body, r.Err)
}

// Extract will get the Snapshot object out of the CreateResult object.
func (r CreateResult) Extract() (*Snapshot, error) {
	return commonExtract(r.Body, r.Err)
}

// Extract will get the Snapshot object out of the UpdateResult object.
func (r UpdateResult) Extract() (*Snapshot, error) {
	return commonExtract(r.Body, r.Err)
}

// ExtractSnapshots extracts and returns Snapshots. It is used while iterating over a snapshots.List call.
func ExtractSnapshots(page pagination.Page) ([]Snapshot, error) {
	var response struct {
		Snapshots []Snapshot `json:"snapshots"`
	}

	err := mapstructure.Decode(page.(os.ListResult).Body, &response)
	return response.Snapshots, err
}

// WaitUntilComplete will continually poll a snapshot until it successfully
// transitions to a specified state. It will do this for at most the number of
// seconds specified.
func (snapshot Snapshot) WaitUntilComplete(c *gophercloud.ServiceClient, timeout int) error {
	return gophercloud.WaitFor(timeout, func() (bool, error) {
		// Poll resource
		current, err := Get(c, snapshot.ID).Extract()
		if err != nil {
			return false, err
		}

		// Has it been built yet?
		if current.Progress == "100%" {
			return true, nil
		}

		return false, nil
	})
}

// WaitUntilDeleted will continually poll a snapshot until it has been
// successfully deleted, i.e. returns a 404 status.
func (snapshot Snapshot) WaitUntilDeleted(c *gophercloud.ServiceClient, timeout int) error {
	return gophercloud.WaitFor(timeout, func() (bool, error) {
		// Poll resource
		_, err := Get(c, snapshot.ID).Extract()

		// Check for a 404
		if casted, ok := err.(*gophercloud.UnexpectedResponseCodeError); ok && casted.Actual == 404 {
			return true, nil
		} else if err != nil {
			return false, err
		}

		return false, nil
	})
}
