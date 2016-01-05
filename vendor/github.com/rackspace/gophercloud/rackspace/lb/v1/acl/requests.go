package acl

import (
	"errors"
	"fmt"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// List is the operation responsible for returning a paginated collection of
// network items that define a load balancer's access list.
func List(client *gophercloud.ServiceClient, lbID int) pagination.Pager {
	url := rootURL(client, lbID)

	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return AccessListPage{pagination.SinglePageBase(r)}
	})
}

// CreateOptsBuilder is the interface responsible for generating the JSON
// for a Create operation.
type CreateOptsBuilder interface {
	ToAccessListCreateMap() (map[string]interface{}, error)
}

// CreateOpts is a slice of CreateOpt structs, that allow the user to create
// multiple nodes in a single operation (one node per CreateOpt).
type CreateOpts []CreateOpt

// CreateOpt represents the options to create a single node.
type CreateOpt struct {
	// Required - the IP address or CIDR for item to add to access list.
	Address string

	// Required - the type of the node. Either ALLOW or DENY.
	Type Type
}

// ToAccessListCreateMap converts a slice of options into a map that can be
// used for the JSON.
func (opts CreateOpts) ToAccessListCreateMap() (map[string]interface{}, error) {
	type itemMap map[string]interface{}
	items := []itemMap{}

	for k, v := range opts {
		if v.Address == "" {
			return itemMap{}, fmt.Errorf("Address is a required attribute, none provided for %d CreateOpt element", k)
		}
		if v.Type != ALLOW && v.Type != DENY {
			return itemMap{}, fmt.Errorf("Type must be ALLOW or DENY")
		}

		item := make(itemMap)
		item["address"] = v.Address
		item["type"] = v.Type

		items = append(items, item)
	}

	return itemMap{"accessList": items}, nil
}

// Create is the operation responsible for adding network items to the access
// rules for a particular load balancer. If network items already exist, the
// new item will be appended. A single IP address or subnet range is considered
// unique and cannot be duplicated.
func Create(client *gophercloud.ServiceClient, loadBalancerID int, opts CreateOptsBuilder) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToAccessListCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = client.Post(rootURL(client, loadBalancerID), reqBody, nil, nil)
	return res
}

// BulkDelete will delete multiple network items from a load balancer's access
// list in a single operation.
func BulkDelete(c *gophercloud.ServiceClient, loadBalancerID int, itemIDs []int) DeleteResult {
	var res DeleteResult

	if len(itemIDs) > 10 || len(itemIDs) == 0 {
		res.Err = errors.New("You must provide a minimum of 1 and a maximum of 10 item IDs")
		return res
	}

	url := rootURL(c, loadBalancerID)
	url += gophercloud.IDSliceToQueryString("id", itemIDs)

	_, res.Err = c.Delete(url, nil)
	return res
}

// Delete will remove a single network item from a load balancer's access list.
func Delete(c *gophercloud.ServiceClient, lbID, itemID int) DeleteResult {
	var res DeleteResult
	_, res.Err = c.Delete(resourceURL(c, lbID, itemID), nil)
	return res
}

// DeleteAll will delete the entire contents of a load balancer's access list,
// effectively resetting it and allowing all traffic.
func DeleteAll(c *gophercloud.ServiceClient, lbID int) DeleteResult {
	var res DeleteResult
	_, res.Err = c.Delete(rootURL(c, lbID), nil)
	return res
}
