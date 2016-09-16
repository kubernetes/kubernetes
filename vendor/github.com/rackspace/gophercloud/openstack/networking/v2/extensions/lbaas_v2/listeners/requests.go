package listeners

import (
	"fmt"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// AdminState gives users a solid type to work with for create and update
// operations. It is recommended that users use the `Up` and `Down` enums.
type AdminState *bool

type listenerOpts struct {
	// Required. The protocol - can either be TCP, HTTP or HTTPS.
	Protocol Protocol

	// Required. The port on which to listen for client traffic.
	ProtocolPort int

	// Required for admins. Indicates the owner of the Listener.
	TenantID string

	// Required. The load balancer on which to provision this listener.
	LoadbalancerID string

	// Human-readable name for the Listener. Does not have to be unique.
	Name string

	// Optional. The ID of the default pool with which the Listener is associated.
	DefaultPoolID string

	// Optional. Human-readable description for the Listener.
	Description string

	// Optional. The maximum number of connections allowed for the Listener.
	ConnLimit *int

	// Optional. A reference to a container of TLS secrets.
	DefaultTlsContainerRef string

	// Optional. A list of references to TLS secrets.
	SniContainerRefs []string

	// Optional. The administrative state of the Listener. A valid value is true (UP)
	// or false (DOWN).
	AdminStateUp *bool
}

// Convenience vars for AdminStateUp values.
var (
	iTrue  = true
	iFalse = false

	Up   AdminState = &iTrue
	Down AdminState = &iFalse
)

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToListenerListQuery() (string, error)
}

// ListOpts allows the filtering and sorting of paginated collections through
// the API. Filtering is achieved by passing in struct field values that map to
// the floating IP attributes you want to see returned. SortKey allows you to
// sort by a particular listener attribute. SortDir sets the direction, and is
// either `asc' or `desc'. Marker and Limit are used for pagination.
type ListOpts struct {
	ID              string `q:"id"`
	Name            string `q:"name"`
	AdminStateUp    *bool  `q:"admin_state_up"`
	TenantID        string `q:"tenant_id"`
	LoadbalancerID  string `q:"loadbalancer_id"`
	DefaultPoolID   string `q:"default_pool_id"`
	Protocol        string `q:"protocol"`
	ProtocolPort    int    `q:"protocol_port"`
	ConnectionLimit int    `q:"connection_limit"`
	Limit           int    `q:"limit"`
	Marker          string `q:"marker"`
	SortKey         string `q:"sort_key"`
	SortDir         string `q:"sort_dir"`
}

// ToListenerListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToListenerListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}
	return q.String(), nil
}

// List returns a Pager which allows you to iterate over a collection of
// routers. It accepts a ListOpts struct, which allows you to filter and sort
// the returned collection for greater efficiency.
//
// Default policy settings return only those routers that are owned by the
// tenant who submits the request, unless an admin user submits the request.
func List(c *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := rootURL(c)
	if opts != nil {
		query, err := opts.ToListenerListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	return pagination.NewPager(c, url, func(r pagination.PageResult) pagination.Page {
		return ListenerPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

type Protocol string

// Supported attributes for create/update operations.
const (
	ProtocolTCP   Protocol = "TCP"
	ProtocolHTTP  Protocol = "HTTP"
	ProtocolHTTPS Protocol = "HTTPS"
)

var (
	errLoadbalancerIdRequired = fmt.Errorf("LoadbalancerID is required")
	errProtocolRequired       = fmt.Errorf("Protocol is required")
	errProtocolPortRequired   = fmt.Errorf("ProtocolPort  is required")
)

// CreateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the main Create operation in this package. Since many
// extensions decorate or modify the common logic, it is useful for them to
// satisfy a basic interface in order for them to be used.
type CreateOptsBuilder interface {
	ToListenerCreateMap() (map[string]interface{}, error)
}

// CreateOpts is the common options struct used in this package's Create
// operation.
type CreateOpts listenerOpts

// ToListenerCreateMap casts a CreateOpts struct to a map.
func (opts CreateOpts) ToListenerCreateMap() (map[string]interface{}, error) {
	l := make(map[string]interface{})

	if opts.LoadbalancerID != "" {
		l["loadbalancer_id"] = opts.LoadbalancerID
	} else {
		return nil, errLoadbalancerIdRequired
	}
	if opts.Protocol != "" {
		l["protocol"] = opts.Protocol
	} else {
		return nil, errProtocolRequired
	}
	if opts.ProtocolPort != 0 {
		l["protocol_port"] = opts.ProtocolPort
	} else {
		return nil, errProtocolPortRequired
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
	if opts.DefaultPoolID != "" {
		l["default_pool_id"] = opts.DefaultPoolID
	}
	if opts.Description != "" {
		l["description"] = opts.Description
	}
	if opts.ConnLimit != nil {
		l["connection_limit"] = &opts.ConnLimit
	}
	if opts.DefaultTlsContainerRef != "" {
		l["default_tls_container_ref"] = opts.DefaultTlsContainerRef
	}
	if opts.SniContainerRefs != nil {
		l["sni_container_refs"] = opts.SniContainerRefs
	}

	return map[string]interface{}{"listener": l}, nil
}

// Create is an operation which provisions a new Listeners based on the
// configuration defined in the CreateOpts struct. Once the request is
// validated and progress has started on the provisioning process, a
// CreateResult will be returned.
//
// Users with an admin role can create Listeners on behalf of other tenants by
// specifying a TenantID attribute different than their own.
func Create(c *gophercloud.ServiceClient, opts CreateOpts) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToListenerCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	// Send request to API
	_, res.Err = c.Post(rootURL(c), reqBody, &res.Body, nil)
	return res
}

// Get retrieves a particular Listeners based on its unique ID.
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
	ToListenerUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts is the common options struct used in this package's Update
// operation.
type UpdateOpts listenerOpts

// ToListenerUpdateMap casts a UpdateOpts struct to a map.
func (opts UpdateOpts) ToListenerUpdateMap() (map[string]interface{}, error) {
	l := make(map[string]interface{})

	if opts.Name != "" {
		l["name"] = opts.Name
	}
	if opts.Description != "" {
		l["description"] = opts.Description
	}
	if opts.ConnLimit != nil {
		l["connection_limit"] = &opts.ConnLimit
	}
	if opts.DefaultTlsContainerRef != "" {
		l["default_tls_container_ref"] = opts.DefaultTlsContainerRef
	}
	if opts.SniContainerRefs != nil {
		l["sni_container_refs"] = opts.SniContainerRefs
	}
	if opts.AdminStateUp != nil {
		l["admin_state_up"] = &opts.AdminStateUp
	}

	return map[string]interface{}{"listener": l}, nil
}

// Update is an operation which modifies the attributes of the specified Listener.
func Update(c *gophercloud.ServiceClient, id string, opts UpdateOpts) UpdateResult {
	var res UpdateResult

	reqBody, err := opts.ToListenerUpdateMap()
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

// Delete will permanently delete a particular Listeners based on its unique ID.
func Delete(c *gophercloud.ServiceClient, id string) DeleteResult {
	var res DeleteResult
	_, res.Err = c.Delete(resourceURL(c, id), nil)
	return res
}
