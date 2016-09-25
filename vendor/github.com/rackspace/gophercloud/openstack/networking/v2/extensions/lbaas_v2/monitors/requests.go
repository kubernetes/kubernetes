package monitors

import (
	"fmt"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

type monitorOpts struct {
	// Required. The Pool to Monitor.
	PoolID string

	// Optional. The Name of the Monitor.
	Name string

	// Required for admins. Indicates the owner of the Loadbalancer.
	TenantID string

	// Required. The type of probe, which is PING, TCP, HTTP, or HTTPS, that is
	// sent by the load balancer to verify the member state.
	Type string

	// Required. The time, in seconds, between sending probes to members.
	Delay int

	// Required. Maximum number of seconds for a Monitor to wait for a ping reply
	// before it times out. The value must be less than the delay value.
	Timeout int

	// Required. Number of permissible ping failures before changing the member's
	// status to INACTIVE. Must be a number between 1 and 10.
	MaxRetries int

	// Required for HTTP(S) types. URI path that will be accessed if Monitor type
	// is HTTP or HTTPS.
	URLPath string

	// Required for HTTP(S) types. The HTTP method used for requests by the
	// Monitor. If this attribute is not specified, it defaults to "GET".
	HTTPMethod string

	// Required for HTTP(S) types. Expected HTTP codes for a passing HTTP(S)
	// Monitor. You can either specify a single status like "200", or a range
	// like "200-202".
	ExpectedCodes string

	AdminStateUp *bool
}

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToMonitorListQuery() (string, error)
}

// ListOpts allows the filtering and sorting of paginated collections through
// the API. Filtering is achieved by passing in struct field values that map to
// the Monitor attributes you want to see returned. SortKey allows you to
// sort by a particular Monitor attribute. SortDir sets the direction, and is
// either `asc' or `desc'. Marker and Limit are used for pagination.
type ListOpts struct {
	ID            string `q:"id"`
	Name          string `q:"name"`
	TenantID      string `q:"tenant_id"`
	PoolID        string `q:"pool_id"`
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

// ToMonitorListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToMonitorListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}
	return q.String(), nil
}

// List returns a Pager which allows you to iterate over a collection of
// health monitors. It accepts a ListOpts struct, which allows you to filter and sort
// the returned collection for greater efficiency.
//
// Default policy settings return only those health monitors that are owned by the
// tenant who submits the request, unless an admin user submits the request.
func List(c *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := rootURL(c)
	if opts != nil {
		query, err := opts.ToMonitorListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	return pagination.NewPager(c, url, func(r pagination.PageResult) pagination.Page {
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
	errPoolIDRequired        = fmt.Errorf("PoolID to monitor is required")
	errValidTypeRequired     = fmt.Errorf("A valid Type is required. Supported values are PING, TCP, HTTP and HTTPS")
	errDelayRequired         = fmt.Errorf("Delay is required")
	errTimeoutRequired       = fmt.Errorf("Timeout is required")
	errMaxRetriesRequired    = fmt.Errorf("MaxRetries is required")
	errURLPathRequired       = fmt.Errorf("URL path is required")
	errExpectedCodesRequired = fmt.Errorf("ExpectedCodes is required")
	errDelayMustGETimeout    = fmt.Errorf("Delay must be greater than or equal to timeout")
)

// CreateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the main Create operation in this package. Since many
// extensions decorate or modify the common logic, it is useful for them to
// satisfy a basic interface in order for them to be used.
type CreateOptsBuilder interface {
	ToMonitorCreateMap() (map[string]interface{}, error)
}

// CreateOpts is the common options struct used in this package's Create
// operation.
type CreateOpts monitorOpts

// ToMonitorCreateMap casts a CreateOpts struct to a map.
func (opts CreateOpts) ToMonitorCreateMap() (map[string]interface{}, error) {
	l := make(map[string]interface{})
	allowed := map[string]bool{TypeHTTP: true, TypeHTTPS: true, TypeTCP: true, TypePING: true}

	if allowed[opts.Type] {
		l["type"] = opts.Type
	} else {
		return nil, errValidTypeRequired
	}
	if opts.Type == TypeHTTP || opts.Type == TypeHTTPS {
		if opts.URLPath != "" {
			l["url_path"] = opts.URLPath
		} else {
			return nil, errURLPathRequired
		}
		if opts.ExpectedCodes != "" {
			l["expected_codes"] = opts.ExpectedCodes
		} else {
			return nil, errExpectedCodesRequired
		}
	}
	if opts.PoolID != "" {
		l["pool_id"] = opts.PoolID
	} else {
		return nil, errPoolIDRequired
	}
	if opts.Delay != 0 {
		l["delay"] = opts.Delay
	} else {
		return nil, errDelayRequired
	}
	if opts.Timeout != 0 {
		l["timeout"] = opts.Timeout
	} else {
		return nil, errMaxRetriesRequired
	}
	if opts.MaxRetries != 0 {
		l["max_retries"] = opts.MaxRetries
	} else {
		return nil, errMaxRetriesRequired
	}
	if opts.AdminStateUp != nil {
		l["admin_state_up"] = &opts.AdminStateUp
	}
	if opts.Name != "" {
		l["name"] = opts.Name
	}
	if opts.TenantID != "" {
		l["tenant_id"] = opts.TenantID
	}
	if opts.HTTPMethod != "" {
		l["http_method"] = opts.HTTPMethod
	}

	return map[string]interface{}{"healthmonitor": l}, nil
}

/*
 Create is an operation which provisions a new Health Monitor. There are
 different types of Monitor you can provision: PING, TCP or HTTP(S). Below
 are examples of how to create each one.

 Here is an example config struct to use when creating a PING or TCP Monitor:

 CreateOpts{Type: TypePING, Delay: 20, Timeout: 10, MaxRetries: 3}
 CreateOpts{Type: TypeTCP, Delay: 20, Timeout: 10, MaxRetries: 3}

 Here is an example config struct to use when creating a HTTP(S) Monitor:

 CreateOpts{Type: TypeHTTP, Delay: 20, Timeout: 10, MaxRetries: 3,
 HttpMethod: "HEAD", ExpectedCodes: "200", PoolID: "2c946bfc-1804-43ab-a2ff-58f6a762b505"}
*/

func Create(c *gophercloud.ServiceClient, opts CreateOptsBuilder) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToMonitorCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	// Send request to API
	_, res.Err = c.Post(rootURL(c), reqBody, &res.Body, nil)
	return res
}

// Get retrieves a particular Health Monitor based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) GetResult {
	var res GetResult
	_, res.Err = c.Get(resourceURL(c, id), &res.Body, nil)
	return res
}

// UpdateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the main Update operation in this package. Since many
// extensions decorate or modify the common logic, it is useful for them to
// satisfy a basic interface in order for them to be used.
type UpdateOptsBuilder interface {
	ToMonitorUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts is the common options struct used in this package's Update
// operation.
type UpdateOpts monitorOpts

// ToMonitorUpdateMap casts a UpdateOpts struct to a map.
func (opts UpdateOpts) ToMonitorUpdateMap() (map[string]interface{}, error) {
	l := make(map[string]interface{})

	if opts.URLPath != "" {
		l["url_path"] = opts.URLPath
	}
	if opts.ExpectedCodes != "" {
		l["expected_codes"] = opts.ExpectedCodes
	}
	if opts.Delay != 0 {
		l["delay"] = opts.Delay
	}
	if opts.Timeout != 0 {
		l["timeout"] = opts.Timeout
	}
	if opts.MaxRetries != 0 {
		l["max_retries"] = opts.MaxRetries
	}
	if opts.AdminStateUp != nil {
		l["admin_state_up"] = &opts.AdminStateUp
	}
	if opts.Name != "" {
		l["name"] = opts.Name
	}
	if opts.HTTPMethod != "" {
		l["http_method"] = opts.HTTPMethod
	}

	return map[string]interface{}{"healthmonitor": l}, nil
}

// Update is an operation which modifies the attributes of the specified Monitor.
func Update(c *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) UpdateResult {
	var res UpdateResult

	reqBody, err := opts.ToMonitorUpdateMap()
	if err != nil {
		res.Err = err
		return res
	}

	// Send request to API
	_, res.Err = c.Put(resourceURL(c, id), reqBody, &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 202},
	})

	return res
}

// Delete will permanently delete a particular Monitor based on its unique ID.
func Delete(c *gophercloud.ServiceClient, id string) DeleteResult {
	var res DeleteResult
	_, res.Err = c.Delete(resourceURL(c, id), nil)
	return res
}
