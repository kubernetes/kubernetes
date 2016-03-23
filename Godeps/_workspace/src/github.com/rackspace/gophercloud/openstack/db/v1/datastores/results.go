package datastores

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// Version represents a version API resource. Multiple versions belong to a Datastore.
type Version struct {
	ID    string
	Links []gophercloud.Link
	Name  string
}

// Datastore represents a Datastore API resource.
type Datastore struct {
	DefaultVersion string `json:"default_version" mapstructure:"default_version"`
	ID             string
	Links          []gophercloud.Link
	Name           string
	Versions       []Version
}

// DatastorePartial is a meta structure which is used in various API responses.
// It is a lightweight and truncated version of a full Datastore resource,
// offering details of the Version, Type and VersionID only.
type DatastorePartial struct {
	Version   string
	Type      string
	VersionID string `json:"version_id" mapstructure:"version_id"`
}

// GetResult represents the result of a Get operation.
type GetResult struct {
	gophercloud.Result
}

// GetVersionResult represents the result of getting a version.
type GetVersionResult struct {
	gophercloud.Result
}

// DatastorePage represents a page of datastore resources.
type DatastorePage struct {
	pagination.SinglePageBase
}

// IsEmpty indicates whether a Datastore collection is empty.
func (r DatastorePage) IsEmpty() (bool, error) {
	is, err := ExtractDatastores(r)
	if err != nil {
		return true, err
	}
	return len(is) == 0, nil
}

// ExtractDatastores retrieves a slice of datastore structs from a paginated
// collection.
func ExtractDatastores(page pagination.Page) ([]Datastore, error) {
	casted := page.(DatastorePage).Body

	var resp struct {
		Datastores []Datastore `mapstructure:"datastores" json:"datastores"`
	}

	err := mapstructure.Decode(casted, &resp)
	return resp.Datastores, err
}

// Extract retrieves a single Datastore struct from an operation result.
func (r GetResult) Extract() (*Datastore, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var response struct {
		Datastore Datastore `mapstructure:"datastore"`
	}

	err := mapstructure.Decode(r.Body, &response)
	return &response.Datastore, err
}

// DatastorePage represents a page of version resources.
type VersionPage struct {
	pagination.SinglePageBase
}

// IsEmpty indicates whether a collection of version resources is empty.
func (r VersionPage) IsEmpty() (bool, error) {
	is, err := ExtractVersions(r)
	if err != nil {
		return true, err
	}
	return len(is) == 0, nil
}

// ExtractVersions retrieves a slice of versions from a paginated collection.
func ExtractVersions(page pagination.Page) ([]Version, error) {
	casted := page.(VersionPage).Body

	var resp struct {
		Versions []Version `mapstructure:"versions" json:"versions"`
	}

	err := mapstructure.Decode(casted, &resp)
	return resp.Versions, err
}

// Extract retrieves a single Version struct from an operation result.
func (r GetVersionResult) Extract() (*Version, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var response struct {
		Version Version `mapstructure:"version"`
	}

	err := mapstructure.Decode(r.Body, &response)
	return &response.Version, err
}
