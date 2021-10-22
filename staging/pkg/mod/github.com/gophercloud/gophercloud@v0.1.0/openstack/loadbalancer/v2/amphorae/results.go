package amphorae

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Amphora is virtual machine, container, dedicated hardware, appliance or device that actually performs the task of
// load balancing in the Octavia system.
type Amphora struct {
	// The unique ID for the Amphora.
	ID string `json:"id"`

	// The ID of the load balancer.
	LoadbalancerID string `json:"loadbalancer_id"`

	// The management IP of the amphora.
	LBNetworkIP string `json:"lb_network_ip"`

	// The ID of the amphora resource in the compute system.
	ComputeID string `json:"compute_id"`

	// The IP address of the Virtual IP (VIP).
	HAIP string `json:"ha_ip"`

	// The ID of the Virtual IP (VIP) port.
	HAPortID string `json:"ha_port_id"`

	// The date the certificate for the amphora expires.
	CertExpiration time.Time `json:"-"`

	// Whether the certificate is in the process of being replaced.
	CertBusy bool `json:"cert_busy"`

	// The role of the amphora. One of STANDALONE, MASTER, BACKUP.
	Role string `json:"role"`

	// The status of the amphora. One of: BOOTING, ALLOCATED, READY, PENDING_CREATE, PENDING_DELETE, DELETED, ERROR.
	Status string `json:"status"`

	// The vrrp port’s ID in the networking system.
	VRRPPortID string `json:"vrrp_port_id"`

	// The address of the vrrp port on the amphora.
	VRRPIP string `json:"vrrp_ip"`

	// The bound interface name of the vrrp port on the amphora.
	VRRPInterface string `json:"vrrp_interface"`

	// The vrrp group’s ID for the amphora.
	VRRPID int `json:"vrrp_id"`

	// The priority of the amphora in the vrrp group.
	VRRPPriority int `json:"vrrp_priority"`

	// The availability zone of a compute instance, cached at create time. This is not guaranteed to be current. May be
	// an empty-string if the compute service does not use zones.
	CachedZone string `json:"cached_zone"`

	// The ID of the glance image used for the amphora.
	ImageID string `json:"image_id"`

	// The UTC date and timestamp when the resource was created.
	CreatedAt time.Time `json:"-"`

	// The UTC date and timestamp when the resource was last updated.
	UpdatedAt time.Time `json:"-"`
}

func (a *Amphora) UnmarshalJSON(b []byte) error {
	type tmp Amphora
	var s struct {
		tmp
		CertExpiration gophercloud.JSONRFC3339NoZ `json:"cert_expiration"`
		CreatedAt      gophercloud.JSONRFC3339NoZ `json:"created_at"`
		UpdatedAt      gophercloud.JSONRFC3339NoZ `json:"updated_at"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*a = Amphora(s.tmp)

	a.CreatedAt = time.Time(s.CreatedAt)
	a.UpdatedAt = time.Time(s.UpdatedAt)
	a.CertExpiration = time.Time(s.CertExpiration)

	return nil
}

// AmphoraPage is the page returned by a pager when traversing over a
// collection of amphorae.
type AmphoraPage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of amphoraes has
// reached the end of a page and the pager seeks to traverse over a new one.
// In order to do this, it needs to construct the next page's URL.
func (r AmphoraPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"amphorae_links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// IsEmpty checks whether a AmphoraPage struct is empty.
func (r AmphoraPage) IsEmpty() (bool, error) {
	is, err := ExtractAmphorae(r)
	return len(is) == 0, err
}

// ExtractAmphorae accepts a Page struct, specifically a AmphoraPage
// struct, and extracts the elements into a slice of Amphora structs. In
// other words, a generic collection is mapped into a relevant slice.
func ExtractAmphorae(r pagination.Page) ([]Amphora, error) {
	var s struct {
		Amphorae []Amphora `json:"amphorae"`
	}
	err := (r.(AmphoraPage)).ExtractInto(&s)
	return s.Amphorae, err
}

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts an amphora.
func (r commonResult) Extract() (*Amphora, error) {
	var s struct {
		Amphora *Amphora `json:"amphora"`
	}
	err := r.ExtractInto(&s)
	return s.Amphora, err
}

// GetResult represents the result of a get operation. Call its Extract
// method to interpret it as an amphora.
type GetResult struct {
	commonResult
}
