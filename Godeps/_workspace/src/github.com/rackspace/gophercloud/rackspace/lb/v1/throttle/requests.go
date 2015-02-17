package throttle

import (
	"errors"

	"github.com/racker/perigee"

	"github.com/rackspace/gophercloud"
)

// CreateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the main Create operation in this package.
type CreateOptsBuilder interface {
	ToCTCreateMap() (map[string]interface{}, error)
}

// CreateOpts is the common options struct used in this package's Create
// operation.
type CreateOpts struct {
	// Required - the maximum amount of connections per IP address to allow per LB.
	MaxConnections int

	// Deprecated as of v1.22.
	MaxConnectionRate int

	// Deprecated as of v1.22.
	MinConnections int

	// Deprecated as of v1.22.
	RateInterval int
}

// ToCTCreateMap casts a CreateOpts struct to a map.
func (opts CreateOpts) ToCTCreateMap() (map[string]interface{}, error) {
	ct := make(map[string]interface{})

	if opts.MaxConnections < 0 || opts.MaxConnections > 100000 {
		return ct, errors.New("MaxConnections must be an int between 0 and 100000")
	}

	ct["maxConnections"] = opts.MaxConnections
	ct["maxConnectionRate"] = opts.MaxConnectionRate
	ct["minConnections"] = opts.MinConnections
	ct["rateInterval"] = opts.RateInterval

	return map[string]interface{}{"connectionThrottle": ct}, nil
}

// Create is the operation responsible for creating or updating the connection
// throttling configuration for a load balancer.
func Create(c *gophercloud.ServiceClient, lbID int, opts CreateOptsBuilder) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToCTCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = perigee.Request("PUT", rootURL(c, lbID), perigee.Options{
		MoreHeaders: c.AuthenticatedHeaders(),
		ReqBody:     &reqBody,
		Results:     &res.Body,
		OkCodes:     []int{202},
	})

	return res
}

// Get is the operation responsible for showing the details of the connection
// throttling configuration for a load balancer.
func Get(c *gophercloud.ServiceClient, lbID int) GetResult {
	var res GetResult

	_, res.Err = perigee.Request("GET", rootURL(c, lbID), perigee.Options{
		MoreHeaders: c.AuthenticatedHeaders(),
		Results:     &res.Body,
		OkCodes:     []int{200},
	})

	return res
}

// Delete is the operation responsible for deleting the connection throttling
// configuration for a load balancer.
func Delete(c *gophercloud.ServiceClient, lbID int) DeleteResult {
	var res DeleteResult

	_, res.Err = perigee.Request("DELETE", rootURL(c, lbID), perigee.Options{
		MoreHeaders: c.AuthenticatedHeaders(),
		OkCodes:     []int{202},
	})

	return res
}
