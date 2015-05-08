package vips

import (
	"errors"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// List is the operation responsible for returning a paginated collection of
// load balancer virtual IP addresses.
func List(client *gophercloud.ServiceClient, loadBalancerID int) pagination.Pager {
	url := rootURL(client, loadBalancerID)
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return VIPPage{pagination.SinglePageBase(r)}
	})
}

// CreateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the main Create operation in this package. Since many
// extensions decorate or modify the common logic, it is useful for them to
// satisfy a basic interface in order for them to be used.
type CreateOptsBuilder interface {
	ToVIPCreateMap() (map[string]interface{}, error)
}

// CreateOpts is the common options struct used in this package's Create
// operation.
type CreateOpts struct {
	// Optional - the ID of an existing virtual IP. By doing this, you are
	// allowing load balancers to share IPV6 addresses.
	ID string

	// Optional - the type of address.
	Type Type

	// Optional - the version of address.
	Version Version
}

// ToVIPCreateMap casts a CreateOpts struct to a map.
func (opts CreateOpts) ToVIPCreateMap() (map[string]interface{}, error) {
	lb := make(map[string]interface{})

	if opts.ID != "" {
		lb["id"] = opts.ID
	}
	if opts.Type != "" {
		lb["type"] = opts.Type
	}
	if opts.Version != "" {
		lb["ipVersion"] = opts.Version
	}

	return lb, nil
}

// Create is the operation responsible for assigning a new Virtual IP to an
// existing load balancer resource. Currently, only version 6 IP addresses may
// be added.
func Create(c *gophercloud.ServiceClient, lbID int, opts CreateOptsBuilder) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToVIPCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = c.Post(rootURL(c, lbID), reqBody, &res.Body, nil)
	return res
}

// BulkDelete is the operation responsible for batch deleting multiple VIPs in
// a single operation. It accepts a slice of integer IDs and will remove them
// from the load balancer. The maximum limit is 10 VIP removals at once.
func BulkDelete(c *gophercloud.ServiceClient, loadBalancerID int, vipIDs []int) DeleteResult {
	var res DeleteResult

	if len(vipIDs) > 10 || len(vipIDs) == 0 {
		res.Err = errors.New("You must provide a minimum of 1 and a maximum of 10 VIP IDs")
		return res
	}

	url := rootURL(c, loadBalancerID)
	url += gophercloud.IDSliceToQueryString("id", vipIDs)

	_, res.Err = c.Delete(url, nil)
	return res
}

// Delete is the operation responsible for permanently deleting a VIP.
func Delete(c *gophercloud.ServiceClient, lbID, vipID int) DeleteResult {
	var res DeleteResult
	_, res.Err = c.Delete(resourceURL(c, lbID, vipID), nil)
	return res
}
