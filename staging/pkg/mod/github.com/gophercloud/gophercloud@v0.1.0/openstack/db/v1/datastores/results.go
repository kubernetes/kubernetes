package datastores

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Version represents a version API resource. Multiple versions belong to a Datastore.
type Version struct {
	ID    string
	Links []gophercloud.Link
	Name  string
}

// Datastore represents a Datastore API resource.
type Datastore struct {
	DefaultVersion string `json:"default_version"`
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
	VersionID string `json:"version_id"`
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
	return len(is) == 0, err
}

// ExtractDatastores retrieves a slice of datastore structs from a paginated
// collection.
func ExtractDatastores(r pagination.Page) ([]Datastore, error) {
	var s struct {
		Datastores []Datastore `json:"datastores"`
	}
	err := (r.(DatastorePage)).ExtractInto(&s)
	return s.Datastores, err
}

// Extract retrieves a single Datastore struct from an operation result.
func (r GetResult) Extract() (*Datastore, error) {
	var s struct {
		Datastore *Datastore `json:"datastore"`
	}
	err := r.ExtractInto(&s)
	return s.Datastore, err
}

// VersionPage represents a page of version resources.
type VersionPage struct {
	pagination.SinglePageBase
}

// IsEmpty indicates whether a collection of version resources is empty.
func (r VersionPage) IsEmpty() (bool, error) {
	is, err := ExtractVersions(r)
	return len(is) == 0, err
}

// ExtractVersions retrieves a slice of versions from a paginated collection.
func ExtractVersions(r pagination.Page) ([]Version, error) {
	var s struct {
		Versions []Version `json:"versions"`
	}
	err := (r.(VersionPage)).ExtractInto(&s)
	return s.Versions, err
}

// Extract retrieves a single Version struct from an operation result.
func (r GetVersionResult) Extract() (*Version, error) {
	var s struct {
		Version *Version `json:"version"`
	}
	err := r.ExtractInto(&s)
	return s.Version, err
}
