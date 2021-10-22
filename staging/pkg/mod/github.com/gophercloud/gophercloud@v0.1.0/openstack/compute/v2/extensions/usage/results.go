package usage

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// TenantUsage is a set of usage information about a tenant over the sampling window
type TenantUsage struct {
	// ServerUsages is an array of ServerUsage maps
	ServerUsages []ServerUsage `json:"server_usages"`

	// Start is the beginning time to calculate usage statistics on compute and storage resources
	Start time.Time `json:"-"`

	// Stop is the ending time to calculate usage statistics on compute and storage resources
	Stop time.Time `json:"-"`

	// TenantID is the ID of the tenant whose usage is being reported on
	TenantID string `json:"tenant_id"`

	// TotalHours is the total duration that servers exist (in hours)
	TotalHours float64 `json:"total_hours"`

	// TotalLocalGBUsage multiplies the server disk size (in GiB) by hours the server exists, and then adding that all together for each server
	TotalLocalGBUsage float64 `json:"total_local_gb_usage"`

	// TotalMemoryMBUsage multiplies the server memory size (in MB) by hours the server exists, and then adding that all together for each server
	TotalMemoryMBUsage float64 `json:"total_memory_mb_usage"`

	// TotalVCPUsUsage multiplies the number of virtual CPUs of the server by hours the server exists, and then adding that all together for each server
	TotalVCPUsUsage float64 `json:"total_vcpus_usage"`
}

// UnmarshalJSON sets *u to a copy of data.
func (u *TenantUsage) UnmarshalJSON(b []byte) error {
	type tmp TenantUsage
	var s struct {
		tmp
		Start gophercloud.JSONRFC3339MilliNoZ `json:"start"`
		Stop  gophercloud.JSONRFC3339MilliNoZ `json:"stop"`
	}

	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	*u = TenantUsage(s.tmp)

	u.Start = time.Time(s.Start)
	u.Stop = time.Time(s.Stop)

	return nil
}

// ServerUsage is a detailed set of information about a specific instance inside a tenant
type ServerUsage struct {
	// EndedAt is the date and time when the server was deleted
	EndedAt time.Time `json:"-"`

	// Flavor is the display name of a flavor
	Flavor string `json:"flavor"`

	// Hours is the duration that the server exists in hours
	Hours float64 `json:"hours"`

	// InstanceID is the UUID of the instance
	InstanceID string `json:"instance_id"`

	// LocalGB is the sum of the root disk size of the server and the ephemeral disk size of it (in GiB)
	LocalGB int `json:"local_gb"`

	// MemoryMB is the memory size of the server (in MB)
	MemoryMB int `json:"memory_mb"`

	// Name is the name assigned to the server when it was created
	Name string `json:"name"`

	// StartedAt is the date and time when the server was started
	StartedAt time.Time `json:"-"`

	// State is the VM power state
	State string `json:"state"`

	// TenantID is the UUID of the tenant in a multi-tenancy cloud
	TenantID string `json:"tenant_id"`

	// Uptime is the uptime of the server in seconds
	Uptime int `json:"uptime"`

	// VCPUs is the number of virtual CPUs that the server uses
	VCPUs int `json:"vcpus"`
}

// UnmarshalJSON sets *u to a copy of data.
func (u *ServerUsage) UnmarshalJSON(b []byte) error {
	type tmp ServerUsage
	var s struct {
		tmp
		EndedAt   gophercloud.JSONRFC3339MilliNoZ `json:"ended_at"`
		StartedAt gophercloud.JSONRFC3339MilliNoZ `json:"started_at"`
	}

	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	*u = ServerUsage(s.tmp)

	u.EndedAt = time.Time(s.EndedAt)
	u.StartedAt = time.Time(s.StartedAt)

	return nil
}

// SingleTenantPage stores a single, only page of TenantUsage results from a
// SingleTenant call.
type SingleTenantPage struct {
	pagination.LinkedPageBase
}

// IsEmpty determines whether or not a SingleTenantPage is empty.
func (r SingleTenantPage) IsEmpty() (bool, error) {
	ks, err := ExtractSingleTenant(r)
	return ks == nil, err
}

// NextPageURL uses the response's embedded link reference to navigate to the
// next page of results.
func (r SingleTenantPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"tenant_usage_links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// ExtractSingleTenant interprets a SingleTenantPage as a TenantUsage result.
func ExtractSingleTenant(page pagination.Page) (*TenantUsage, error) {
	var s struct {
		TenantUsage *TenantUsage `json:"tenant_usage"`
	}
	err := (page.(SingleTenantPage)).ExtractInto(&s)
	return s.TenantUsage, err
}

// AllTenantsPage stores a single, only page of TenantUsage results from a
// AllTenants call.
type AllTenantsPage struct {
	pagination.LinkedPageBase
}

// ExtractAllTenants interprets a AllTenantsPage as a TenantUsage result.
func ExtractAllTenants(page pagination.Page) ([]TenantUsage, error) {
	var s struct {
		TenantUsages []TenantUsage `json:"tenant_usages"`
	}
	err := (page.(AllTenantsPage)).ExtractInto(&s)
	return s.TenantUsages, err
}

// IsEmpty determines whether or not an AllTenantsPage is empty.
func (r AllTenantsPage) IsEmpty() (bool, error) {
	usages, err := ExtractAllTenants(r)
	return len(usages) == 0, err
}

// NextPageURL uses the response's embedded link reference to navigate to the
// next page of results.
func (r AllTenantsPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"tenant_usages_links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}
