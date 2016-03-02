package floatingip

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// A FloatingIP is an IP that can be associated with an instance
type FloatingIP struct {
	// ID is a unique ID of the Floating IP
	ID string `mapstructure:"id"`

	// FixedIP is the IP of the instance related to the Floating IP
	FixedIP string `mapstructure:"fixed_ip,omitempty"`

	// InstanceID is the ID of the instance that is using the Floating IP
	InstanceID string `mapstructure:"instance_id"`

	// IP is the actual Floating IP
	IP string `mapstructure:"ip"`

	// Pool is the pool of floating IPs that this floating IP belongs to
	Pool string `mapstructure:"pool"`
}

// FloatingIPsPage stores a single, only page of FloatingIPs
// results from a List call.
type FloatingIPsPage struct {
	pagination.SinglePageBase
}

// IsEmpty determines whether or not a FloatingIPsPage is empty.
func (page FloatingIPsPage) IsEmpty() (bool, error) {
	va, err := ExtractFloatingIPs(page)
	return len(va) == 0, err
}

// ExtractFloatingIPs interprets a page of results as a slice of
// FloatingIPs.
func ExtractFloatingIPs(page pagination.Page) ([]FloatingIP, error) {
	casted := page.(FloatingIPsPage).Body
	var response struct {
		FloatingIPs []FloatingIP `mapstructure:"floating_ips"`
	}

	err := mapstructure.WeakDecode(casted, &response)

	return response.FloatingIPs, err
}

type FloatingIPResult struct {
	gophercloud.Result
}

// Extract is a method that attempts to interpret any FloatingIP resource
// response as a FloatingIP struct.
func (r FloatingIPResult) Extract() (*FloatingIP, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		FloatingIP *FloatingIP `json:"floating_ip" mapstructure:"floating_ip"`
	}

	err := mapstructure.WeakDecode(r.Body, &res)
	return res.FloatingIP, err
}

// CreateResult is the response from a Create operation. Call its Extract method to interpret it
// as a FloatingIP.
type CreateResult struct {
	FloatingIPResult
}

// GetResult is the response from a Get operation. Call its Extract method to interpret it
// as a FloatingIP.
type GetResult struct {
	FloatingIPResult
}

// DeleteResult is the response from a Delete operation. Call its Extract method to determine if
// the call succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// AssociateResult is the response from a Delete operation. Call its Extract method to determine if
// the call succeeded or failed.
type AssociateResult struct {
	gophercloud.ErrResult
}

// DisassociateResult is the response from a Delete operation. Call its Extract method to determine if
// the call succeeded or failed.
type DisassociateResult struct {
	gophercloud.ErrResult
}
