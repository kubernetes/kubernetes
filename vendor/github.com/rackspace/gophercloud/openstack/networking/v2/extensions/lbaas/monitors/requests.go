package monitors

import (
	"fmt"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// ListOpts allows the filtering and sorting of paginated collections through
// the API. Filtering is achieved by passing in struct field values that map to
// the floating IP attributes you want to see returned. SortKey allows you to
// sort by a particular network attribute. SortDir sets the direction, and is
// either `asc' or `desc'. Marker and Limit are used for pagination.
type ListOpts struct {
	ID            string `q:"id"`
	TenantID      string `q:"tenant_id"`
	Type          string `q:"type"`
	Delay         int    `q:"delay"`
	Timeout       int    `q:"timeout"`
	MaxRetries    int    `q:"max_retries"`
	HTTPMethod    string `q:"http_method"`
	URLPath       string `q:"url_path"`
	ExpectedCodes string `q:"expected_codes"`
	AdminStateUp  *bool  `q:"admin_state_up"`
	Status        string `q:"status"`
	Limit         int    `q:"limit"`
	Marker        string `q:"marker"`
	SortKey       string `q:"sort_key"`
	SortDir       string `q:"sort_dir"`
}

// List returns a Pager which allows you to iterate over a collection of
// routers. It accepts a ListOpts struct, which allows you to filter and sort
// the returned collection for greater efficiency.
//
// Default policy settings return only those routers that are owned by the
// tenant who submits the request, unless an admin user submits the request.
func List(c *gophercloud.ServiceClient, opts ListOpts) pagination.Pager {
	q, err := gophercloud.BuildQueryString(&opts)
	if err != nil {
		return pagination.Pager{Err: err}
	}
	u := rootURL(c) + q.String()

	return pagination.NewPager(c, u, func(r pagination.PageResult) pagination.Page {
		return MonitorPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// Constants that represent approved monitoring types.
const (
	TypePING  = "PING"
	TypeTCP   = "TCP"
	TypeHTTP  = "HTTP"
	TypeHTTPS = "HTTPS"
)

var (
	errValidTypeRequired     = fmt.Errorf("A valid Type is required. Supported values are PING, TCP, HTTP and HTTPS")
	errDelayRequired         = fmt.Errorf("Delay is required")
	errTimeoutRequired       = fmt.Errorf("Timeout is required")
	errMaxRetriesRequired    = fmt.Errorf("MaxRetries is required")
	errURLPathRequired       = fmt.Errorf("URL path is required")
	errExpectedCodesRequired = fmt.Errorf("ExpectedCodes is required")
	errDelayMustGETimeout    = fmt.Errorf("Delay must be greater than or equal to timeout")
)

// CreateOpts contains all the values needed to create a new health monitor.
type CreateOpts struct {
	// Required for admins. Indicates the owner of the VIP.
	TenantID string

	// Required. The type of probe, which is PING, TCP, HTTP, or HTTPS, that is
	// sent by the load balancer to verify the member state.
	Type string

	// Required. The time, in seconds, between sending probes to members.
	Delay int

	// Required. Maximum number of seconds for a monitor to wait for a ping reply
	// before it times out. The value must be less than the delay value.
	Timeout int

	// Required. Number of permissible ping failures before changing the member's
	// status to INACTIVE. Must be a number between 1 and 10.
	MaxRetries int

	// Required for HTTP(S) types. URI path that will be accessed if monitor type
	// is HTTP or HTTPS.
	URLPath string

	// Required for HTTP(S) types. The HTTP method used for requests by the
	// monitor. If this attribute is not specified, it defaults to "GET".
	HTTPMethod string

	// Required for HTTP(S) types. Expected HTTP codes for a passing HTTP(S)
	// monitor. You can either specify a single status like "200", or a range
	// like "200-202".
	ExpectedCodes string

	AdminStateUp *bool
}

// Create is an operation which provisions a new health monitor. There are
// different types of monitor you can provision: PING, TCP or HTTP(S). Below
// are examples of how to create each one.
//
// Here is an example config struct to use when creating a PING or TCP monitor:
//
// CreateOpts{Type: TypePING, Delay: 20, Timeout: 10, MaxRetries: 3}
// CreateOpts{Type: TypeTCP, Delay: 20, Timeout: 10, MaxRetries: 3}
//
// Here is an example config struct to use when creating a HTTP(S) monitor:
//
// CreateOpts{Type: TypeHTTP, Delay: 20, Timeout: 10, MaxRetries: 3,
//  HttpMethod: "HEAD", ExpectedCodes: "200"}
//
func Create(c *gophercloud.ServiceClient, opts CreateOpts) CreateResult {
	var res CreateResult

	// Validate inputs
	allowed := map[string]bool{TypeHTTP: true, TypeHTTPS: true, TypeTCP: true, TypePING: true}
	if opts.Type == "" || allowed[opts.Type] == false {
		res.Err = errValidTypeRequired
	}
	if opts.Delay == 0 {
		res.Err = errDelayRequired
	}
	if opts.Timeout == 0 {
		res.Err = errTimeoutRequired
	}
	if opts.MaxRetries == 0 {
		res.Err = errMaxRetriesRequired
	}
	if opts.Type == TypeHTTP || opts.Type == TypeHTTPS {
		if opts.URLPath == "" {
			res.Err = errURLPathRequired
		}
		if opts.ExpectedCodes == "" {
			res.Err = errExpectedCodesRequired
		}
	}
	if opts.Delay < opts.Timeout {
		res.Err = errDelayMustGETimeout
	}
	if res.Err != nil {
		return res
	}

	type monitor struct {
		Type          string  `json:"type"`
		Delay         int     `json:"delay"`
		Timeout       int     `json:"timeout"`
		MaxRetries    int     `json:"max_retries"`
		TenantID      *string `json:"tenant_id,omitempty"`
		URLPath       *string `json:"url_path,omitempty"`
		ExpectedCodes *string `json:"expected_codes,omitempty"`
		HTTPMethod    *string `json:"http_method,omitempty"`
		AdminStateUp  *bool   `json:"admin_state_up,omitempty"`
	}

	type request struct {
		Monitor monitor `json:"health_monitor"`
	}

	reqBody := request{Monitor: monitor{
		Type:          opts.Type,
		Delay:         opts.Delay,
		Timeout:       opts.Timeout,
		MaxRetries:    opts.MaxRetries,
		TenantID:      gophercloud.MaybeString(opts.TenantID),
		URLPath:       gophercloud.MaybeString(opts.URLPath),
		ExpectedCodes: gophercloud.MaybeString(opts.ExpectedCodes),
		HTTPMethod:    gophercloud.MaybeString(opts.HTTPMethod),
		AdminStateUp:  opts.AdminStateUp,
	}}

	_, res.Err = c.Post(rootURL(c), reqBody, &res.Body, nil)
	return res
}

// Get retrieves a particular health monitor based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) GetResult {
	var res GetResult
	_, res.Err = c.Get(resourceURL(c, id), &res.Body, nil)
	return res
}

// UpdateOpts contains all the values needed to update an existing virtual IP.
// Attributes not listed here but appear in CreateOpts are immutable and cannot
// be updated.
type UpdateOpts struct {
	// Required. The time, in seconds, between sending probes to members.
	Delay int

	// Required. Maximum number of seconds for a monitor to wait for a ping reply
	// before it times out. The value must be less than the delay value.
	Timeout int

	// Required. Number of permissible ping failures before changing the member's
	// status to INACTIVE. Must be a number between 1 and 10.
	MaxRetries int

	// Required for HTTP(S) types. URI path that will be accessed if monitor type
	// is HTTP or HTTPS.
	URLPath string

	// Required for HTTP(S) types. The HTTP method used for requests by the
	// monitor. If this attribute is not specified, it defaults to "GET".
	HTTPMethod string

	// Required for HTTP(S) types. Expected HTTP codes for a passing HTTP(S)
	// monitor. You can either specify a single status like "200", or a range
	// like "200-202".
	ExpectedCodes string

	AdminStateUp *bool
}

// Update is an operation which modifies the attributes of the specified monitor.
func Update(c *gophercloud.ServiceClient, id string, opts UpdateOpts) UpdateResult {
	var res UpdateResult

	if opts.Delay > 0 && opts.Timeout > 0 && opts.Delay < opts.Timeout {
		res.Err = errDelayMustGETimeout
	}

	type monitor struct {
		Delay         int     `json:"delay"`
		Timeout       int     `json:"timeout"`
		MaxRetries    int     `json:"max_retries"`
		URLPath       *string `json:"url_path,omitempty"`
		ExpectedCodes *string `json:"expected_codes,omitempty"`
		HTTPMethod    *string `json:"http_method,omitempty"`
		AdminStateUp  *bool   `json:"admin_state_up,omitempty"`
	}

	type request struct {
		Monitor monitor `json:"health_monitor"`
	}

	reqBody := request{Monitor: monitor{
		Delay:         opts.Delay,
		Timeout:       opts.Timeout,
		MaxRetries:    opts.MaxRetries,
		URLPath:       gophercloud.MaybeString(opts.URLPath),
		ExpectedCodes: gophercloud.MaybeString(opts.ExpectedCodes),
		HTTPMethod:    gophercloud.MaybeString(opts.HTTPMethod),
		AdminStateUp:  opts.AdminStateUp,
	}}

	_, res.Err = c.Put(resourceURL(c, id), reqBody, &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 202},
	})

	return res
}

// Delete will permanently delete a particular monitor based on its unique ID.
func Delete(c *gophercloud.ServiceClient, id string) DeleteResult {
	var res DeleteResult
	_, res.Err = c.Delete(resourceURL(c, id), nil)
	return res
}
