package configurations

import (
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Config represents a configuration group API resource.
type Config struct {
	Created              time.Time `json:"created"`
	Updated              time.Time `json:"updated"`
	DatastoreName        string    `json:"datastore_name"`
	DatastoreVersionID   string    `json:"datastore_version_id"`
	DatastoreVersionName string    `json:"datastore_version_name"`
	Description          string
	ID                   string
	Name                 string
	Values               map[string]interface{}
}

// ConfigPage contains a page of Config resources in a paginated collection.
type ConfigPage struct {
	pagination.SinglePageBase
}

// IsEmpty indicates whether a ConfigPage is empty.
func (r ConfigPage) IsEmpty() (bool, error) {
	is, err := ExtractConfigs(r)
	return len(is) == 0, err
}

// ExtractConfigs will retrieve a slice of Config structs from a page.
func ExtractConfigs(r pagination.Page) ([]Config, error) {
	var s struct {
		Configs []Config `json:"configurations"`
	}
	err := (r.(ConfigPage)).ExtractInto(&s)
	return s.Configs, err
}

type commonResult struct {
	gophercloud.Result
}

// Extract will retrieve a Config resource from an operation result.
func (r commonResult) Extract() (*Config, error) {
	var s struct {
		Config *Config `json:"configuration"`
	}
	err := r.ExtractInto(&s)
	return s.Config, err
}

// GetResult represents the result of a Get operation.
type GetResult struct {
	commonResult
}

// CreateResult represents the result of a Create operation.
type CreateResult struct {
	commonResult
}

// UpdateResult represents the result of an Update operation.
type UpdateResult struct {
	gophercloud.ErrResult
}

// ReplaceResult represents the result of a Replace operation.
type ReplaceResult struct {
	gophercloud.ErrResult
}

// DeleteResult represents the result of a Delete operation.
type DeleteResult struct {
	gophercloud.ErrResult
}

// Param represents a configuration parameter API resource.
type Param struct {
	Max             float64
	Min             float64
	Name            string
	RestartRequired bool `json:"restart_required"`
	Type            string
}

// ParamPage contains a page of Param resources in a paginated collection.
type ParamPage struct {
	pagination.SinglePageBase
}

// IsEmpty indicates whether a ParamPage is empty.
func (r ParamPage) IsEmpty() (bool, error) {
	is, err := ExtractParams(r)
	return len(is) == 0, err
}

// ExtractParams will retrieve a slice of Param structs from a page.
func ExtractParams(r pagination.Page) ([]Param, error) {
	var s struct {
		Params []Param `json:"configuration-parameters"`
	}
	err := (r.(ParamPage)).ExtractInto(&s)
	return s.Params, err
}

// ParamResult represents the result of an operation which retrieves details
// about a particular configuration param.
type ParamResult struct {
	gophercloud.Result
}

// Extract will retrieve a param from an operation result.
func (r ParamResult) Extract() (*Param, error) {
	var s *Param
	err := r.ExtractInto(&s)
	return s, err
}
