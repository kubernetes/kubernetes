package networks

import (
	"errors"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// List returns a Pager which allows you to iterate over a collection of
// networks. It accepts a ListOpts struct, which allows you to filter and sort
// the returned collection for greater efficiency.
func List(c *gophercloud.ServiceClient) pagination.Pager {
	createPage := func(r pagination.PageResult) pagination.Page {
		return NetworkPage{pagination.SinglePageBase(r)}
	}

	return pagination.NewPager(c, listURL(c), createPage)
}

// Get retrieves a specific network based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) GetResult {
	var res GetResult
	_, res.Err = c.Get(getURL(c, id), &res.Body, nil)
	return res
}

// CreateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the main Create operation in this package. Since many
// extensions decorate or modify the common logic, it is useful for them to
// satisfy a basic interface in order for them to be used.
type CreateOptsBuilder interface {
	ToNetworkCreateMap() (map[string]interface{}, error)
}

// CreateOpts is the common options struct used in this package's Create
// operation.
type CreateOpts struct {
	// REQUIRED. See Network object for more info.
	CIDR string
	// REQUIRED. See Network object for more info.
	Label string
}

// ToNetworkCreateMap casts a CreateOpts struct to a map.
func (opts CreateOpts) ToNetworkCreateMap() (map[string]interface{}, error) {
	n := make(map[string]interface{})

	if opts.CIDR == "" {
		return nil, errors.New("Required field CIDR not set.")
	}
	if opts.Label == "" {
		return nil, errors.New("Required field Label not set.")
	}

	n["label"] = opts.Label
	n["cidr"] = opts.CIDR
	return map[string]interface{}{"network": n}, nil
}

// Create accepts a CreateOpts struct and creates a new network using the values
// provided. This operation does not actually require a request body, i.e. the
// CreateOpts struct argument can be empty.
//
// The tenant ID that is contained in the URI is the tenant that creates the
// network. An admin user, however, has the option of specifying another tenant
// ID in the CreateOpts struct.
func Create(c *gophercloud.ServiceClient, opts CreateOptsBuilder) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToNetworkCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	// Send request to API
	_, res.Err = c.Post(createURL(c), reqBody, &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201, 202},
	})
	return res
}

// Delete accepts a unique ID and deletes the network associated with it.
func Delete(c *gophercloud.ServiceClient, networkID string) DeleteResult {
	var res DeleteResult
	_, res.Err = c.Delete(deleteURL(c, networkID), nil)
	return res
}
