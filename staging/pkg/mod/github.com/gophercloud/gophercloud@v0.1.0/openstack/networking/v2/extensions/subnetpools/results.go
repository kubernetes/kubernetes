package subnetpools

import (
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a subnetpool resource.
func (r commonResult) Extract() (*SubnetPool, error) {
	var s struct {
		SubnetPool *SubnetPool `json:"subnetpool"`
	}
	err := r.ExtractInto(&s)
	return s.SubnetPool, err
}

// GetResult represents the result of a get operation. Call its Extract
// method to interpret it as a SubnetPool.
type GetResult struct {
	commonResult
}

// CreateResult represents the result of a create operation. Call its Extract
// method to interpret it as a SubnetPool.
type CreateResult struct {
	commonResult
}

// UpdateResult represents the result of an update operation. Call its Extract
// method to interpret it as a SubnetPool.
type UpdateResult struct {
	commonResult
}

// DeleteResult represents the result of a delete operation. Call its
// ExtractErr method to determine if the request succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// SubnetPool represents a Neutron subnetpool.
// A subnetpool is a pool of addresses from which subnets can be allocated.
type SubnetPool struct {
	// ID is the id of the subnetpool.
	ID string `json:"id"`

	// Name is the human-readable name of the subnetpool.
	Name string `json:"name"`

	// DefaultQuota is the per-project quota on the prefix space
	// that can be allocated from the subnetpool for project subnets.
	DefaultQuota int `json:"default_quota"`

	// TenantID is the id of the Identity project.
	TenantID string `json:"tenant_id"`

	// ProjectID is the id of the Identity project.
	ProjectID string `json:"project_id"`

	// CreatedAt is the time at which subnetpool has been created.
	CreatedAt time.Time `json:"created_at"`

	// UpdatedAt is the time at which subnetpool has been created.
	UpdatedAt time.Time `json:"updated_at"`

	// Prefixes is the list of subnet prefixes to assign to the subnetpool.
	// Neutron API merges adjacent prefixes and treats them as a single prefix.
	// Each subnet prefix must be unique among all subnet prefixes in all subnetpools
	// that are associated with the address scope.
	Prefixes []string `json:"prefixes"`

	// DefaultPrefixLen is yhe size of the prefix to allocate when the cidr
	// or prefixlen attributes are omitted when you create the subnet.
	// Defaults to the MinPrefixLen.
	DefaultPrefixLen int `json:"-"`

	// MinPrefixLen is the smallest prefix that can be allocated from a subnetpool.
	// For IPv4 subnetpools, default is 8.
	// For IPv6 subnetpools, default is 64.
	MinPrefixLen int `json:"-"`

	// MaxPrefixLen is the maximum prefix size that can be allocated from the subnetpool.
	// For IPv4 subnetpools, default is 32.
	// For IPv6 subnetpools, default is 128.
	MaxPrefixLen int `json:"-"`

	// AddressScopeID is the Neutron address scope to assign to the subnetpool.
	AddressScopeID string `json:"address_scope_id"`

	// IPversion is the IP protocol version.
	// Valid value is 4 or 6. Default is 4.
	IPversion int `json:"ip_version"`

	// Shared indicates whether this network is shared across all projects.
	Shared bool `json:"shared"`

	// Description is thehuman-readable description for the resource.
	Description string `json:"description"`

	// IsDefault indicates if the subnetpool is default pool or not.
	IsDefault bool `json:"is_default"`

	// RevisionNumber is the revision number of the subnetpool.
	RevisionNumber int `json:"revision_number"`

	// Tags optionally set via extensions/attributestags
	Tags []string `json:"tags"`
}

func (r *SubnetPool) UnmarshalJSON(b []byte) error {
	type tmp SubnetPool
	var s struct {
		tmp
		DefaultPrefixLen interface{} `json:"default_prefixlen"`
		MinPrefixLen     interface{} `json:"min_prefixlen"`
		MaxPrefixLen     interface{} `json:"max_prefixlen"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}

	*r = SubnetPool(s.tmp)

	switch t := s.DefaultPrefixLen.(type) {
	case string:
		if r.DefaultPrefixLen, err = strconv.Atoi(t); err != nil {
			return err
		}
	case float64:
		r.DefaultPrefixLen = int(t)
	default:
		return fmt.Errorf("DefaultPrefixLen has unexpected type: %T", t)
	}

	switch t := s.MinPrefixLen.(type) {
	case string:
		if r.MinPrefixLen, err = strconv.Atoi(t); err != nil {
			return err
		}
	case float64:
		r.MinPrefixLen = int(t)
	default:
		return fmt.Errorf("MinPrefixLen has unexpected type: %T", t)
	}

	switch t := s.MaxPrefixLen.(type) {
	case string:
		if r.MaxPrefixLen, err = strconv.Atoi(t); err != nil {
			return err
		}
	case float64:
		r.MaxPrefixLen = int(t)
	default:
		return fmt.Errorf("MaxPrefixLen has unexpected type: %T", t)
	}

	return nil
}

// SubnetPoolPage stores a single page of SubnetPools from a List() API call.
type SubnetPoolPage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of subnetpools has reached
// the end of a page and the pager seeks to traverse over a new one.
// In order to do this, it needs to construct the next page's URL.
func (r SubnetPoolPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"subnetpools_links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// IsEmpty determines whether or not a SubnetPoolPage is empty.
func (r SubnetPoolPage) IsEmpty() (bool, error) {
	subnetpools, err := ExtractSubnetPools(r)
	return len(subnetpools) == 0, err
}

// ExtractSubnetPools interprets the results of a single page from a List() API call,
// producing a slice of SubnetPools structs.
func ExtractSubnetPools(r pagination.Page) ([]SubnetPool, error) {
	var s struct {
		SubnetPools []SubnetPool `json:"subnetpools"`
	}
	err := (r.(SubnetPoolPage)).ExtractInto(&s)
	return s.SubnetPools, err
}
