package services

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Service is a VPN Service
type Service struct {
	// TenantID is the ID of the project.
	TenantID string `json:"tenant_id"`

	// ProjectID is the ID of the project.
	ProjectID string `json:"project_id"`

	// SubnetID is the ID of the subnet.
	SubnetID string `json:"subnet_id"`

	// RouterID is the ID of the router.
	RouterID string `json:"router_id"`

	// Description is a human-readable description for the resource.
	// Default is an empty string
	Description string `json:"description"`

	// AdminStateUp is the administrative state of the resource, which is up (true) or down (false).
	AdminStateUp bool `json:"admin_state_up"`

	// Name is the human readable name of the service.
	Name string `json:"name"`

	// Status indicates whether IPsec VPN service is currently operational.
	// Values are ACTIVE, DOWN, BUILD, ERROR, PENDING_CREATE, PENDING_UPDATE, or PENDING_DELETE.
	Status string `json:"status"`

	// ID is the unique ID of the VPN service.
	ID string `json:"id"`

	// ExternalV6IP is the read-only external (public) IPv6 address that is used for the VPN service.
	ExternalV6IP string `json:"external_v6_ip"`

	// ExternalV4IP is the read-only external (public) IPv4 address that is used for the VPN service.
	ExternalV4IP string `json:"external_v4_ip"`

	// FlavorID is the ID of the flavor.
	FlavorID string `json:"flavor_id"`
}

type commonResult struct {
	gophercloud.Result
}

// ServicePage is the page returned by a pager when traversing over a
// collection of VPN services.
type ServicePage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of VPN services has
// reached the end of a page and the pager seeks to traverse over a new one.
// In order to do this, it needs to construct the next page's URL.
func (r ServicePage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"vpnservices_links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// IsEmpty checks whether a ServicePage struct is empty.
func (r ServicePage) IsEmpty() (bool, error) {
	is, err := ExtractServices(r)
	return len(is) == 0, err
}

// ExtractServices accepts a Page struct, specifically a Service struct,
// and extracts the elements into a slice of Service structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractServices(r pagination.Page) ([]Service, error) {
	var s struct {
		Services []Service `json:"vpnservices"`
	}
	err := (r.(ServicePage)).ExtractInto(&s)
	return s.Services, err
}

// GetResult represents the result of a get operation. Call its Extract
// method to interpret it as a Service.
type GetResult struct {
	commonResult
}

// Extract is a function that accepts a result and extracts a VPN service.
func (r commonResult) Extract() (*Service, error) {
	var s struct {
		Service *Service `json:"vpnservice"`
	}
	err := r.ExtractInto(&s)
	return s.Service, err
}

// CreateResult represents the result of a create operation. Call its Extract
// method to interpret it as a Service.
type CreateResult struct {
	commonResult
}

// DeleteResult represents the result of a delete operation. Call its
// ExtractErr method to determine if the operation succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// UpdateResult represents the result of an update operation. Call its Extract
// method to interpret it as a service.
type UpdateResult struct {
	commonResult
}
