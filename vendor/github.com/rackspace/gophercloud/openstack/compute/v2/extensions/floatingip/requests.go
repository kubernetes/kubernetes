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

// AssociateOpts specifies the required information to associate or disassociate a floating IP to an instance
type AssociateOpts struct {
	// ServerID is the UUID of the server
	ServerID string

	// FixedIP is an optional fixed IP address of the server
	FixedIP string

	// FloatingIP is the floating IP to associate with an instance
	FloatingIP string
}

// ToFloatingIPCreateMap constructs a request body from CreateOpts.
func (opts CreateOpts) ToFloatingIPCreateMap() (map[string]interface{}, error) {
	if opts.Pool == "" {
		return nil, errors.New("Missing field required for floating IP creation: Pool")
	}

	return map[string]interface{}{"pool": opts.Pool}, nil
}

// ToAssociateMap constructs a request body from AssociateOpts.
func (opts AssociateOpts) ToAssociateMap() (map[string]interface{}, error) {
	if opts.ServerID == "" {
		return nil, errors.New("Required field missing for floating IP association: ServerID")
	}

	if opts.FloatingIP == "" {
		return nil, errors.New("Required field missing for floating IP association: FloatingIP")
	}

	associateInfo := map[string]interface{}{
		"serverId":   opts.ServerID,
		"floatingIp": opts.FloatingIP,
		"fixedIp":    opts.FixedIP,
	}

	return associateInfo, nil

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
// Deprecated. Use AssociateInstance.
func Associate(client *gophercloud.ServiceClient, serverId, fip string) AssociateResult {
	var res AssociateResult

	addFloatingIp := make(map[string]interface{})
	addFloatingIp["address"] = fip
	reqBody := map[string]interface{}{"addFloatingIp": addFloatingIp}

	_, res.Err = client.Post(associateURL(client, serverId), reqBody, nil, nil)
	return res
}

// AssociateInstance pairs an allocated floating IP with an instance.
func AssociateInstance(client *gophercloud.ServiceClient, opts AssociateOpts) AssociateResult {
	var res AssociateResult

	associateInfo, err := opts.ToAssociateMap()
	if err != nil {
		res.Err = err
		return res
	}

	addFloatingIp := make(map[string]interface{})
	addFloatingIp["address"] = associateInfo["floatingIp"].(string)

	// fixedIp is not required
	if associateInfo["fixedIp"] != "" {
		addFloatingIp["fixed_address"] = associateInfo["fixedIp"].(string)
	}

	serverId := associateInfo["serverId"].(string)

	reqBody := map[string]interface{}{"addFloatingIp": addFloatingIp}
	_, res.Err = client.Post(associateURL(client, serverId), reqBody, nil, nil)
	return res
}

// Disassociate decouples an allocated floating IP from an instance
// Deprecated. Use DisassociateInstance.
func Disassociate(client *gophercloud.ServiceClient, serverId, fip string) DisassociateResult {
	var res DisassociateResult

	removeFloatingIp := make(map[string]interface{})
	removeFloatingIp["address"] = fip
	reqBody := map[string]interface{}{"removeFloatingIp": removeFloatingIp}

	_, res.Err = client.Post(disassociateURL(client, serverId), reqBody, nil, nil)
	return res
}

// DisassociateInstance decouples an allocated floating IP from an instance
func DisassociateInstance(client *gophercloud.ServiceClient, opts AssociateOpts) DisassociateResult {
	var res DisassociateResult

	associateInfo, err := opts.ToAssociateMap()
	if err != nil {
		res.Err = err
		return res
	}

	removeFloatingIp := make(map[string]interface{})
	removeFloatingIp["address"] = associateInfo["floatingIp"].(string)
	reqBody := map[string]interface{}{"removeFloatingIp": removeFloatingIp}

	serverId := associateInfo["serverId"].(string)

	_, res.Err = client.Post(disassociateURL(client, serverId), reqBody, nil, nil)
	return res
}
