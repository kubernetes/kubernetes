package floatingip

import (
	"errors"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// List returns a Pager that allows you to iterate over a collection of FloatingIPs.
func List(client *gophercloud.ServiceClient) pagination.Pager {
	return pagination.NewPager(client, listURL(client), func(r pagination.PageResult) pagination.Page {
		return FloatingIPsPage{pagination.SinglePageBase(r)}
	})
}

// CreateOptsBuilder describes struct types that can be accepted by the Create call. Notable, the
// CreateOpts struct in this package does.
type CreateOptsBuilder interface {
	ToFloatingIPCreateMap() (map[string]interface{}, error)
}

// CreateOpts specifies a Floating IP allocation request
type CreateOpts struct {
	// Pool is the pool of floating IPs to allocate one from
	Pool string
}

// ToFloatingIPCreateMap constructs a request body from CreateOpts.
func (opts CreateOpts) ToFloatingIPCreateMap() (map[string]interface{}, error) {
	if opts.Pool == "" {
		return nil, errors.New("Missing field required for floating IP creation: Pool")
	}

	return map[string]interface{}{"pool": opts.Pool}, nil
}

// Create requests the creation of a new floating IP
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToFloatingIPCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = client.Post(createURL(client), reqBody, &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return res
}

// Get returns data about a previously created FloatingIP.
func Get(client *gophercloud.ServiceClient, id string) GetResult {
	var res GetResult
	_, res.Err = client.Get(getURL(client, id), &res.Body, nil)
	return res
}

// Delete requests the deletion of a previous allocated FloatingIP.
func Delete(client *gophercloud.ServiceClient, id string) DeleteResult {
	var res DeleteResult
	_, res.Err = client.Delete(deleteURL(client, id), nil)
	return res
}

// association / disassociation

// Associate pairs an allocated floating IP with an instance
func Associate(client *gophercloud.ServiceClient, serverId, fip string) AssociateResult {
	var res AssociateResult

	addFloatingIp := make(map[string]interface{})
	addFloatingIp["address"] = fip
	reqBody := map[string]interface{}{"addFloatingIp": addFloatingIp}

	_, res.Err = client.Post(associateURL(client, serverId), reqBody, nil, nil)
	return res
}

// Disassociate decouples an allocated floating IP from an instance
func Disassociate(client *gophercloud.ServiceClient, serverId, fip string) DisassociateResult {
	var res DisassociateResult

	removeFloatingIp := make(map[string]interface{})
	removeFloatingIp["address"] = fip
	reqBody := map[string]interface{}{"removeFloatingIp": removeFloatingIp}

	_, res.Err = client.Post(disassociateURL(client, serverId), reqBody, nil, nil)
	return res
}
