package sessions

import (
	"errors"

	"github.com/rackspace/gophercloud"
)

// CreateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the main Create operation in this package.
type CreateOptsBuilder interface {
	ToSPCreateMap() (map[string]interface{}, error)
}

// CreateOpts is the common options struct used in this package's Create
// operation.
type CreateOpts struct {
	// Required - can either be HTTPCOOKIE or SOURCEIP
	Type Type
}

// ToSPCreateMap casts a CreateOpts struct to a map.
func (opts CreateOpts) ToSPCreateMap() (map[string]interface{}, error) {
	sp := make(map[string]interface{})

	if opts.Type == "" {
		return sp, errors.New("Type is a required field")
	}

	sp["persistenceType"] = opts.Type
	return map[string]interface{}{"sessionPersistence": sp}, nil
}

// Enable is the operation responsible for enabling session persistence for a
// particular load balancer.
func Enable(c *gophercloud.ServiceClient, lbID int, opts CreateOptsBuilder) EnableResult {
	var res EnableResult

	reqBody, err := opts.ToSPCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = c.Put(rootURL(c, lbID), reqBody, &res.Body, nil)
	return res
}

// Get is the operation responsible for showing details of the session
// persistence configuration for a particular load balancer.
func Get(c *gophercloud.ServiceClient, lbID int) GetResult {
	var res GetResult
	_, res.Err = c.Get(rootURL(c, lbID), &res.Body, nil)
	return res
}

// Disable is the operation responsible for disabling session persistence for a
// particular load balancer.
func Disable(c *gophercloud.ServiceClient, lbID int) DisableResult {
	var res DisableResult
	_, res.Err = c.Delete(rootURL(c, lbID), nil)
	return res
}
