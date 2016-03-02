package acl

import (
	"github.com/mitchellh/mapstructure"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// AccessList represents the rules of network access to a particular load
// balancer.
type AccessList []NetworkItem

// NetworkItem describes how an IP address or entire subnet may interact with a
// load balancer.
type NetworkItem struct {
	// The IP address or subnet (CIDR) that defines the network item.
	Address string

	// The numeric unique ID for this item.
	ID int

	// Either ALLOW or DENY.
	Type Type
}

// Type defines how an item may connect to the load balancer.
type Type string

// Convenience consts.
const (
	ALLOW Type = "ALLOW"
	DENY  Type = "DENY"
)

// AccessListPage is the page returned by a pager for traversing over a
// collection of network items in an access list.
type AccessListPage struct {
	pagination.SinglePageBase
}

// IsEmpty checks whether an AccessListPage struct is empty.
func (p AccessListPage) IsEmpty() (bool, error) {
	is, err := ExtractAccessList(p)
	if err != nil {
		return true, nil
	}
	return len(is) == 0, nil
}

// ExtractAccessList accepts a Page struct, specifically an AccessListPage
// struct, and extracts the elements into a slice of NetworkItem structs. In
// other words, a generic collection is mapped into a relevant slice.
func ExtractAccessList(page pagination.Page) (AccessList, error) {
	var resp struct {
		List AccessList `mapstructure:"accessList" json:"accessList"`
	}

	err := mapstructure.Decode(page.(AccessListPage).Body, &resp)

	return resp.List, err
}

// CreateResult represents the result of a create operation.
type CreateResult struct {
	gophercloud.ErrResult
}

// DeleteResult represents the result of a delete operation.
type DeleteResult struct {
	gophercloud.ErrResult
}
