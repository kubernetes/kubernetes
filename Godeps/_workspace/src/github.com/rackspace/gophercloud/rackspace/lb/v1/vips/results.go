package vips

import (
	"github.com/mitchellh/mapstructure"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// VIP represents a Virtual IP API resource.
type VIP struct {
	Address string  `json:"address,omitempty"`
	ID      int     `json:"id,omitempty"`
	Type    Type    `json:"type,omitempty"`
	Version Version `json:"ipVersion,omitempty" mapstructure:"ipVersion"`
}

// Version represents the version of a VIP.
type Version string

// Convenient constants to use for type
const (
	IPV4 Version = "IPV4"
	IPV6 Version = "IPV6"
)

// Type represents the type of a VIP.
type Type string

const (
	// PUBLIC indicates a VIP type that is routable on the public Internet.
	PUBLIC Type = "PUBLIC"

	// PRIVATE indicates a VIP type that is routable only on ServiceNet.
	PRIVATE Type = "SERVICENET"
)

// VIPPage is the page returned by a pager when traversing over a collection
// of VIPs.
type VIPPage struct {
	pagination.SinglePageBase
}

// IsEmpty checks whether a VIPPage struct is empty.
func (p VIPPage) IsEmpty() (bool, error) {
	is, err := ExtractVIPs(p)
	if err != nil {
		return true, nil
	}
	return len(is) == 0, nil
}

// ExtractVIPs accepts a Page struct, specifically a VIPPage struct, and
// extracts the elements into a slice of VIP structs. In other words, a
// generic collection is mapped into a relevant slice.
func ExtractVIPs(page pagination.Page) ([]VIP, error) {
	var resp struct {
		VIPs []VIP `mapstructure:"virtualIps" json:"virtualIps"`
	}

	err := mapstructure.Decode(page.(VIPPage).Body, &resp)

	return resp.VIPs, err
}

type commonResult struct {
	gophercloud.Result
}

func (r commonResult) Extract() (*VIP, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	resp := &VIP{}
	err := mapstructure.Decode(r.Body, resp)

	return resp, err
}

// CreateResult represents the result of a create operation.
type CreateResult struct {
	commonResult
}

// DeleteResult represents the result of a delete operation.
type DeleteResult struct {
	gophercloud.ErrResult
}
