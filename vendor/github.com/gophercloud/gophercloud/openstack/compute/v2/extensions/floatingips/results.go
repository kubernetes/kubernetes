package floatingips

import (
	"encoding/json"
	"strconv"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// A FloatingIP is an IP that can be associated with a server.
type FloatingIP struct {
	// ID is a unique ID of the Floating IP
	ID string `json:"-"`

	// FixedIP is a specific IP on the server to pair the Floating IP with.
	FixedIP string `json:"fixed_ip,omitempty"`

	// InstanceID is the ID of the server that is using the Floating IP.
	InstanceID string `json:"instance_id"`

	// IP is the actual Floating IP.
	IP string `json:"ip"`

	// Pool is the pool of Floating IPs that this Floating IP belongs to.
	Pool string `json:"pool"`
}

func (r *FloatingIP) UnmarshalJSON(b []byte) error {
	type tmp FloatingIP
	var s struct {
		tmp
		ID interface{} `json:"id"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}

	*r = FloatingIP(s.tmp)

	switch t := s.ID.(type) {
	case float64:
		r.ID = strconv.FormatFloat(t, 'f', -1, 64)
	case string:
		r.ID = t
	}

	return err
}

// FloatingIPPage stores a single page of FloatingIPs from a List call.
type FloatingIPPage struct {
	pagination.SinglePageBase
}

// IsEmpty determines whether or not a FloatingIPsPage is empty.
func (page FloatingIPPage) IsEmpty() (bool, error) {
	va, err := ExtractFloatingIPs(page)
	return len(va) == 0, err
}

// ExtractFloatingIPs interprets a page of results as a slice of FloatingIPs.
func ExtractFloatingIPs(r pagination.Page) ([]FloatingIP, error) {
	var s struct {
		FloatingIPs []FloatingIP `json:"floating_ips"`
	}
	err := (r.(FloatingIPPage)).ExtractInto(&s)
	return s.FloatingIPs, err
}

// FloatingIPResult is the raw result from a FloatingIP request.
type FloatingIPResult struct {
	gophercloud.Result
}

// Extract is a method that attempts to interpret any FloatingIP resource
// response as a FloatingIP struct.
func (r FloatingIPResult) Extract() (*FloatingIP, error) {
	var s struct {
		FloatingIP *FloatingIP `json:"floating_ip"`
	}
	err := r.ExtractInto(&s)
	return s.FloatingIP, err
}

// CreateResult is the response from a Create operation. Call its Extract method
// to interpret it as a FloatingIP.
type CreateResult struct {
	FloatingIPResult
}

// GetResult is the response from a Get operation. Call its Extract method to
// interpret it as a FloatingIP.
type GetResult struct {
	FloatingIPResult
}

// DeleteResult is the response from a Delete operation. Call its ExtractErr
// method to determine if the call succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// AssociateResult is the response from a Delete operation. Call its ExtractErr
// method to determine if the call succeeded or failed.
type AssociateResult struct {
	gophercloud.ErrResult
}

// DisassociateResult is the response from a Delete operation. Call its
// ExtractErr method to determine if the call succeeded or failed.
type DisassociateResult struct {
	gophercloud.ErrResult
}
