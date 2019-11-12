package floatingips

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// FloatingIP represents a floating IP resource. A floating IP is an external
// IP address that is mapped to an internal port and, optionally, a specific
// IP address on a private network. In other words, it enables access to an
// instance on a private network from an external network. For this reason,
// floating IPs can only be defined on networks where the `router:external'
// attribute (provided by the external network extension) is set to True.
type FloatingIP struct {
	// ID is the unique identifier for the floating IP instance.
	ID string `json:"id"`

	// Description for the floating IP instance.
	Description string `json:"description"`

	// FloatingNetworkID is the UUID of the external network where the floating
	// IP is to be created.
	FloatingNetworkID string `json:"floating_network_id"`

	// FloatingIP is the address of the floating IP on the external network.
	FloatingIP string `json:"floating_ip_address"`

	// PortID is the UUID of the port on an internal network that is associated
	// with the floating IP.
	PortID string `json:"port_id"`

	// FixedIP is the specific IP address of the internal port which should be
	// associated with the floating IP.
	FixedIP string `json:"fixed_ip_address"`

	// TenantID is the project owner of the floating IP. Only admin users can
	// specify a project identifier other than its own.
	TenantID string `json:"tenant_id"`

	// UpdatedAt and CreatedAt contain ISO-8601 timestamps of when the state of
	// the floating ip last changed, and when it was created.
	UpdatedAt time.Time `json:"-"`
	CreatedAt time.Time `json:"-"`

	// ProjectID is the project owner of the floating IP.
	ProjectID string `json:"project_id"`

	// Status is the condition of the API resource.
	Status string `json:"status"`

	// RouterID is the ID of the router used for this floating IP.
	RouterID string `json:"router_id"`

	// Tags optionally set via extensions/attributestags
	Tags []string `json:"tags"`
}

func (r *FloatingIP) UnmarshalJSON(b []byte) error {
	type tmp FloatingIP

	// Support for older neutron time format
	var s1 struct {
		tmp
		CreatedAt gophercloud.JSONRFC3339NoZ `json:"created_at"`
		UpdatedAt gophercloud.JSONRFC3339NoZ `json:"updated_at"`
	}

	err := json.Unmarshal(b, &s1)
	if err == nil {
		*r = FloatingIP(s1.tmp)
		r.CreatedAt = time.Time(s1.CreatedAt)
		r.UpdatedAt = time.Time(s1.UpdatedAt)

		return nil
	}

	// Support for newer neutron time format
	var s2 struct {
		tmp
		CreatedAt time.Time `json:"created_at"`
		UpdatedAt time.Time `json:"updated_at"`
	}

	err = json.Unmarshal(b, &s2)
	if err != nil {
		return err
	}

	*r = FloatingIP(s2.tmp)
	r.CreatedAt = time.Time(s2.CreatedAt)
	r.UpdatedAt = time.Time(s2.UpdatedAt)

	return nil
}

type commonResult struct {
	gophercloud.Result
}

// Extract will extract a FloatingIP resource from a result.
func (r commonResult) Extract() (*FloatingIP, error) {
	var s FloatingIP
	err := r.ExtractInto(&s)
	return &s, err
}

func (r commonResult) ExtractInto(v interface{}) error {
	return r.Result.ExtractIntoStructPtr(v, "floatingip")
}

// CreateResult represents the result of a create operation. Call its Extract
// method to interpret it as a FloatingIP.
type CreateResult struct {
	commonResult
}

// GetResult represents the result of a get operation. Call its Extract
// method to interpret it as a FloatingIP.
type GetResult struct {
	commonResult
}

// UpdateResult represents the result of an update operation. Call its Extract
// method to interpret it as a FloatingIP.
type UpdateResult struct {
	commonResult
}

// DeleteResult represents the result of an update operation. Call its
// ExtractErr method to determine if the request succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// FloatingIPPage is the page returned by a pager when traversing over a
// collection of floating IPs.
type FloatingIPPage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of floating IPs has
// reached the end of a page and the pager seeks to traverse over a new one.
// In order to do this, it needs to construct the next page's URL.
func (r FloatingIPPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"floatingips_links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// IsEmpty checks whether a FloatingIPPage struct is empty.
func (r FloatingIPPage) IsEmpty() (bool, error) {
	is, err := ExtractFloatingIPs(r)
	return len(is) == 0, err
}

// ExtractFloatingIPs accepts a Page struct, specifically a FloatingIPPage
// struct, and extracts the elements into a slice of FloatingIP structs. In
// other words, a generic collection is mapped into a relevant slice.
func ExtractFloatingIPs(r pagination.Page) ([]FloatingIP, error) {
	var s struct {
		FloatingIPs []FloatingIP `json:"floatingips"`
	}
	err := (r.(FloatingIPPage)).ExtractInto(&s)
	return s.FloatingIPs, err
}

func ExtractFloatingIPsInto(r pagination.Page, v interface{}) error {
	return r.(FloatingIPPage).Result.ExtractIntoSlicePtr(v, "floatingips")
}
