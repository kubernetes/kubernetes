package monitors

import (
	"fmt"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
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

// MonitorType is the type for all the types of LB monitors
type MonitorType string

// Constants that represent approved monitoring types.
const (
	TypePING  MonitorType = "PING"
	TypeTCP   MonitorType = "TCP"
	TypeHTTP  MonitorType = "HTTP"
	TypeHTTPS MonitorType = "HTTPS"
)

// CreateOptsBuilder is what types must satisfy to be used as Create
// options.
type CreateOptsBuilder interface {
	ToLBMonitorCreateMap() (map[string]interface{}, error)
}

// CreateOpts contains all the values needed to create a new health monitor.
type CreateOpts struct {
	// Required. The type of probe, which is PING, TCP, HTTP, or HTTPS, that is
	// sent by the load balancer to verify the member state.
	Type MonitorType `json:"type" required:"true"`
	// Required. The time, in seconds, between sending probes to members.
	Delay int `json:"delay" required:"true"`
	// Required. Maximum number of seconds for a monitor to wait for a ping reply
	// before it times out. The value must be less than the delay value.
	Timeout int `json:"timeout" required:"true"`
	// Required. Number of permissible ping failures before changing the member's
	// status to INACTIVE. Must be a number between 1 and 10.
	MaxRetries int `json:"max_retries" required:"true"`
	// Required for HTTP(S) types. URI path that will be accessed if monitor type
	// is HTTP or HTTPS.
	URLPath string `json:"url_path,omitempty"`
	// Required for HTTP(S) types. The HTTP method used for requests by the
	// monitor. If this attribute is not specified, it defaults to "GET".
	HTTPMethod string `json:"http_method,omitempty"`
	// Required for HTTP(S) types. Expected HTTP codes for a passing HTTP(S)
	// monitor. You can either specify a single status like "200", or a range
	// like "200-202".
	ExpectedCodes string `json:"expected_codes,omitempty"`
	// Required for admins. Indicates the owner of the VIP.
	TenantID     string `json:"tenant_id,omitempty"`
	AdminStateUp *bool  `json:"admin_state_up,omitempty"`
}

// ToLBMonitorCreateMap allows CreateOpts to satisfy the CreateOptsBuilder
// interface
func (opts CreateOpts) ToLBMonitorCreateMap() (map[string]interface{}, error) {
	if opts.Type == TypeHTTP || opts.Type == TypeHTTPS {
		if opts.URLPath == "" {
			err := gophercloud.ErrMissingInput{}
			err.Argument = "monitors.CreateOpts.URLPath"
			return nil, err
		}
		if opts.ExpectedCodes == "" {
			err := gophercloud.ErrMissingInput{}
			err.Argument = "monitors.CreateOpts.ExpectedCodes"
			return nil, err
		}
	}
	if opts.Delay < opts.Timeout {
		err := gophercloud.ErrInvalidInput{}
		err.Argument = "monitors.CreateOpts.Delay/monitors.CreateOpts.Timeout"
		err.Info = "Delay must be greater than or equal to timeout"
		return nil, err
	}
	return gophercloud.BuildRequestBody(opts, "health_monitor")
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
func Create(c *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToLBMonitorCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Post(rootURL(c), b, &r.Body, nil)
	return
}

// Get retrieves a particular health monitor based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = c.Get(resourceURL(c, id), &r.Body, nil)
	return
}

// UpdateOptsBuilder is what types must satisfy to be used as Update
// options.
type UpdateOptsBuilder interface {
	ToLBMonitorUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts contains all the values needed to update an existing virtual IP.
// Attributes not listed here but appear in CreateOpts are immutable and cannot
// be updated.
type UpdateOpts struct {
	// The time, in seconds, between sending probes to members.
	Delay int `json:"delay,omitempty"`
	// Maximum number of seconds for a monitor to wait for a ping reply
	// before it times out. The value must be less than the delay value.
	Timeout int `json:"timeout,omitempty"`
	// Number of permissible ping failures before changing the member's
	// status to INACTIVE. Must be a number between 1 and 10.
	MaxRetries int `json:"max_retries,omitempty"`
	// URI path that will be accessed if monitor type
	// is HTTP or HTTPS.
	URLPath string `json:"url_path,omitempty"`
	// The HTTP method used for requests by the
	// monitor. If this attribute is not specified, it defaults to "GET".
	HTTPMethod string `json:"http_method,omitempty"`
	// Expected HTTP codes for a passing HTTP(S)
	// monitor. You can either specify a single status like "200", or a range
	// like "200-202".
	ExpectedCodes string `json:"expected_codes,omitempty"`
	AdminStateUp  *bool  `json:"admin_state_up,omitempty"`
}

// ToLBMonitorUpdateMap allows UpdateOpts to satisfy the UpdateOptsBuilder
// interface
func (opts UpdateOpts) ToLBMonitorUpdateMap() (map[string]interface{}, error) {
	if opts.Delay > 0 && opts.Timeout > 0 && opts.Delay < opts.Timeout {
		err := gophercloud.ErrInvalidInput{}
		err.Argument = "monitors.CreateOpts.Delay/monitors.CreateOpts.Timeout"
		err.Value = fmt.Sprintf("%d/%d", opts.Delay, opts.Timeout)
		err.Info = "Delay must be greater than or equal to timeout"
		return nil, err
	}
	return gophercloud.BuildRequestBody(opts, "health_monitor")
}

// Update is an operation which modifies the attributes of the specified monitor.
func Update(c *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToLBMonitorUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Put(resourceURL(c, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 202},
	})
	return
}

// Delete will permanently delete a particular monitor based on its unique ID.
func Delete(c *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = c.Delete(resourceURL(c, id), nil)
	return
}
