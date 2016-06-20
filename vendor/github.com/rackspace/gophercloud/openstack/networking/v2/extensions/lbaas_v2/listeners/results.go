package listeners

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/pools"
	"github.com/rackspace/gophercloud/pagination"
)

type LoadBalancerID struct {
	ID string `mapstructure:"id" json:"id"`
}

// Listener is the primary load balancing configuration object that specifies
// the loadbalancer and port on which client traffic is received, as well
// as other details such as the load balancing method to be use, protocol, etc.
type Listener struct {
	// The unique ID for the Listener.
	ID string `mapstructure:"id" json:"id"`

	// Owner of the Listener. Only an admin user can specify a tenant ID other than its own.
	TenantID string `mapstructure:"tenant_id" json:"tenant_id"`

	// Human-readable name for the Listener. Does not have to be unique.
	Name string `mapstructure:"name" json:"name"`

	// Human-readable description for the Listener.
	Description string `mapstructure:"description" json:"description"`

	// The protocol to loadbalance. A valid value is TCP, HTTP, or HTTPS.
	Protocol string `mapstructure:"protocol" json:"protocol"`

	// The port on which to listen to client traffic that is associated with the
	// Loadbalancer. A valid value is from 0 to 65535.
	ProtocolPort int `mapstructure:"protocol_port" json:"protocol_port"`

	// The UUID of default pool. Must have compatible protocol with listener.
	DefaultPoolID string `mapstructure:"default_pool_id" json:"default_pool_id"`

	// A list of load balancer IDs.
	Loadbalancers []LoadBalancerID `mapstructure:"loadbalancers" json:"loadbalancers"`

	// The maximum number of connections allowed for the Loadbalancer. Default is -1,
	// meaning no limit.
	ConnLimit int `mapstructure:"connection_limit" json:"connection_limit"`

	// The list of references to TLS secrets.
	SniContainerRefs []string `mapstructure:"sni_container_refs" json:"sni_container_refs"`

	// Optional. A reference to a container of TLS secrets.
	DefaultTlsContainerRef string `mapstructure:"default_tls_container_ref" json:"default_tls_container_ref"`

	// The administrative state of the Listener. A valid value is true (UP) or false (DOWN).
	AdminStateUp bool `mapstructure:"admin_state_up" json:"admin_state_up"`

	Pools []pools.Pool `mapstructure:"pools" json:"pools"`
}

// ListenerPage is the page returned by a pager when traversing over a
// collection of routers.
type ListenerPage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of routers has reached
// the end of a page and the pager seeks to traverse over a new one. In order
// to do this, it needs to construct the next page's URL.
func (p ListenerPage) NextPageURL() (string, error) {
	type resp struct {
		Links []gophercloud.Link `mapstructure:"listeners_links"`
	}

	var r resp
	err := mapstructure.Decode(p.Body, &r)
	if err != nil {
		return "", err
	}

	return gophercloud.ExtractNextURL(r.Links)
}

// IsEmpty checks whether a RouterPage struct is empty.
func (p ListenerPage) IsEmpty() (bool, error) {
	is, err := ExtractListeners(p)
	if err != nil {
		return true, nil
	}
	return len(is) == 0, nil
}

// ExtractListeners accepts a Page struct, specifically a ListenerPage struct,
// and extracts the elements into a slice of Listener structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractListeners(page pagination.Page) ([]Listener, error) {
	var resp struct {
		Listeners []Listener `mapstructure:"listeners" json:"listeners"`
	}
	err := mapstructure.Decode(page.(ListenerPage).Body, &resp)
	return resp.Listeners, err
}

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts a router.
func (r commonResult) Extract() (*Listener, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		Listener *Listener `mapstructure:"listener" json:"listener"`
	}

	err := mapstructure.Decode(r.Body, &res)

	return res.Listener, err
}

// CreateResult represents the result of a create operation.
type CreateResult struct {
	commonResult
}

// GetResult represents the result of a get operation.
type GetResult struct {
	commonResult
}

// UpdateResult represents the result of an update operation.
type UpdateResult struct {
	commonResult
}

// DeleteResult represents the result of a delete operation.
type DeleteResult struct {
	gophercloud.ErrResult
}
