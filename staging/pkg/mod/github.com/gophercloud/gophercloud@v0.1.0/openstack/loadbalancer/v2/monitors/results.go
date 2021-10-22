package monitors

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

type PoolID struct {
	ID string `json:"id"`
}

// Monitor represents a load balancer health monitor. A health monitor is used
// to determine whether or not back-end members of the VIP's pool are usable
// for processing a request. A pool can have several health monitors associated
// with it. There are different types of health monitors supported:
//
// PING: used to ping the members using ICMP.
// TCP: used to connect to the members using TCP.
// HTTP: used to send an HTTP request to the member.
// HTTPS: used to send a secure HTTP request to the member.
//
// When a pool has several monitors associated with it, each member of the pool
// is monitored by all these monitors. If any monitor declares the member as
// unhealthy, then the member status is changed to INACTIVE and the member
// won't participate in its pool's load balancing. In other words, ALL monitors
// must declare the member to be healthy for it to stay ACTIVE.
type Monitor struct {
	// The unique ID for the Monitor.
	ID string `json:"id"`

	// The Name of the Monitor.
	Name string `json:"name"`

	// The owner of the Monitor.
	ProjectID string `json:"project_id"`

	// The type of probe sent by the load balancer to verify the member state,
	// which is PING, TCP, HTTP, or HTTPS.
	Type string `json:"type"`

	// The time, in seconds, between sending probes to members.
	Delay int `json:"delay"`

	// The maximum number of seconds for a monitor to wait for a connection to be
	// established before it times out. This value must be less than the delay
	// value.
	Timeout int `json:"timeout"`

	// Number of allowed connection failures before changing the status of the
	// member to INACTIVE. A valid value is from 1 to 10.
	MaxRetries int `json:"max_retries"`

	// The HTTP method that the monitor uses for requests.
	HTTPMethod string `json:"http_method"`

	// The HTTP path of the request sent by the monitor to test the health of a
	// member. Must be a string beginning with a forward slash (/).
	URLPath string `json:"url_path" `

	// Expected HTTP codes for a passing HTTP(S) monitor.
	ExpectedCodes string `json:"expected_codes"`

	// The administrative state of the health monitor, which is up (true) or
	// down (false).
	AdminStateUp bool `json:"admin_state_up"`

	// The status of the health monitor. Indicates whether the health monitor is
	// operational.
	Status string `json:"status"`

	// List of pools that are associated with the health monitor.
	Pools []PoolID `json:"pools"`

	// The provisioning status of the Monitor.
	// This value is ACTIVE, PENDING_* or ERROR.
	ProvisioningStatus string `json:"provisioning_status"`

	// The operating status of the monitor.
	OperatingStatus string `json:"operating_status"`
}

// MonitorPage is the page returned by a pager when traversing over a
// collection of health monitors.
type MonitorPage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of monitors has reached
// the end of a page and the pager seeks to traverse over a new one. In order
// to do this, it needs to construct the next page's URL.
func (r MonitorPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"healthmonitors_links"`
	}

	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}

	return gophercloud.ExtractNextURL(s.Links)
}

// IsEmpty checks whether a MonitorPage struct is empty.
func (r MonitorPage) IsEmpty() (bool, error) {
	is, err := ExtractMonitors(r)
	return len(is) == 0, err
}

// ExtractMonitors accepts a Page struct, specifically a MonitorPage struct,
// and extracts the elements into a slice of Monitor structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractMonitors(r pagination.Page) ([]Monitor, error) {
	var s struct {
		Monitors []Monitor `json:"healthmonitors"`
	}
	err := (r.(MonitorPage)).ExtractInto(&s)
	return s.Monitors, err
}

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a monitor.
func (r commonResult) Extract() (*Monitor, error) {
	var s struct {
		Monitor *Monitor `json:"healthmonitor"`
	}
	err := r.ExtractInto(&s)
	return s.Monitor, err
}

// CreateResult represents the result of a create operation. Call its Extract
// method to interpret it as a Monitor.
type CreateResult struct {
	commonResult
}

// GetResult represents the result of a get operation. Call its Extract
// method to interpret it as a Monitor.
type GetResult struct {
	commonResult
}

// UpdateResult represents the result of an update operation. Call its Extract
// method to interpret it as a Monitor.
type UpdateResult struct {
	commonResult
}

// DeleteResult represents the result of a delete operation. Call its
// ExtractErr method to determine if the result succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}
