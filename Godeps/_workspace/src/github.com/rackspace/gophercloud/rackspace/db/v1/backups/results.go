package backups

import (
	"fmt"
	"reflect"
	"time"

	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack/db/v1/datastores"
	"github.com/rackspace/gophercloud/pagination"
)

// Status represents the various states a Backup can be in.
type Status string

// Enum types for the status.
const (
	StatusNew          Status = "NEW"
	StatusBuilding     Status = "BUILDING"
	StatusCompleted    Status = "COMPLETED"
	StatusFailed       Status = "FAILED"
	StatusDeleteFailed Status = "DELETE_FAILED"
)

// Backup represents a Backup API resource.
type Backup struct {
	Description string
	ID          string
	InstanceID  string `json:"instance_id" mapstructure:"instance_id"`
	LocationRef string
	Name        string
	ParentID    string `json:"parent_id" mapstructure:"parent_id"`
	Size        float64
	Status      Status
	Created     time.Time `mapstructure:"-"`
	Updated     time.Time `mapstructure:"-"`
	Datastore   datastores.DatastorePartial
}

// CreateResult represents the result of a create operation.
type CreateResult struct {
	commonResult
}

// GetResult represents the result of a get operation.
type GetResult struct {
	commonResult
}

// DeleteResult represents the result of a delete operation.
type DeleteResult struct {
	gophercloud.ErrResult
}

type commonResult struct {
	gophercloud.Result
}

// Extract will retrieve a Backup struct from an operation's result.
func (r commonResult) Extract() (*Backup, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var response struct {
		Backup Backup `mapstructure:"backup"`
	}

	err := mapstructure.Decode(r.Body, &response)
	val := r.Body.(map[string]interface{})["backup"].(map[string]interface{})

	if t, ok := val["created"].(string); ok && t != "" {
		creationTime, err := time.Parse(time.RFC3339, t)
		if err != nil {
			return &response.Backup, err
		}
		response.Backup.Created = creationTime
	}

	if t, ok := val["updated"].(string); ok && t != "" {
		updatedTime, err := time.Parse(time.RFC3339, t)
		if err != nil {
			return &response.Backup, err
		}
		response.Backup.Updated = updatedTime
	}

	return &response.Backup, err
}

// BackupPage represents a page of backups.
type BackupPage struct {
	pagination.SinglePageBase
}

// IsEmpty checks whether an BackupPage struct is empty.
func (r BackupPage) IsEmpty() (bool, error) {
	is, err := ExtractBackups(r)
	if err != nil {
		return true, err
	}
	return len(is) == 0, nil
}

// ExtractBackups will retrieve a slice of Backup structs from a paginated collection.
func ExtractBackups(page pagination.Page) ([]Backup, error) {
	casted := page.(BackupPage).Body

	var resp struct {
		Backups []Backup `mapstructure:"backups" json:"backups"`
	}

	if err := mapstructure.Decode(casted, &resp); err != nil {
		return nil, err
	}

	var vals []interface{}
	switch casted.(type) {
	case map[string]interface{}:
		vals = casted.(map[string]interface{})["backups"].([]interface{})
	case map[string][]interface{}:
		vals = casted.(map[string][]interface{})["backups"]
	default:
		return resp.Backups, fmt.Errorf("Unknown type: %v", reflect.TypeOf(casted))
	}

	for i, v := range vals {
		val := v.(map[string]interface{})

		if t, ok := val["created"].(string); ok && t != "" {
			creationTime, err := time.Parse(time.RFC3339, t)
			if err != nil {
				return resp.Backups, err
			}
			resp.Backups[i].Created = creationTime
		}

		if t, ok := val["updated"].(string); ok && t != "" {
			updatedTime, err := time.Parse(time.RFC3339, t)
			if err != nil {
				return resp.Backups, err
			}
			resp.Backups[i].Updated = updatedTime
		}
	}

	return resp.Backups, nil
}
