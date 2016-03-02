package lbs

import (
	"errors"

	"github.com/mitchellh/mapstructure"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/lb/v1/acl"
	"github.com/rackspace/gophercloud/rackspace/lb/v1/monitors"
	"github.com/rackspace/gophercloud/rackspace/lb/v1/nodes"
	"github.com/rackspace/gophercloud/rackspace/lb/v1/sessions"
	"github.com/rackspace/gophercloud/rackspace/lb/v1/throttle"
	"github.com/rackspace/gophercloud/rackspace/lb/v1/vips"
)

var (
	errNameRequired    = errors.New("Name is a required attribute")
	errTimeoutExceeded = errors.New("Timeout must be less than 120")
)

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToLBListQuery() (string, error)
}

// ListOpts allows the filtering and sorting of paginated collections through
// the API.
type ListOpts struct {
	ChangesSince string `q:"changes-since"`
	Status       Status `q:"status"`
	NodeAddr     string `q:"nodeaddress"`
	Marker       string `q:"marker"`
	Limit        int    `q:"limit"`
}

// ToLBListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToLBListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}
	return q.String(), nil
}

// List is the operation responsible for returning a paginated collection of
// load balancers. You may pass in a ListOpts struct to filter results.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := rootURL(client)
	if opts != nil {
		query, err := opts.ToLBListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return LBPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// CreateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the main Create operation in this package. Since many
// extensions decorate or modify the common logic, it is useful for them to
// satisfy a basic interface in order for them to be used.
type CreateOptsBuilder interface {
	ToLBCreateMap() (map[string]interface{}, error)
}

// CreateOpts is the common options struct used in this package's Create
// operation.
type CreateOpts struct {
	// Required - name of the load balancer to create. The name must be 128
	// characters or fewer in length, and all UTF-8 characters are valid.
	Name string

	// Optional - nodes to be added.
	Nodes []nodes.Node

	// Required - protocol of the service that is being load balanced.
	// See http://docs.rackspace.com/loadbalancers/api/v1.0/clb-devguide/content/protocols.html
	// for a full list of supported protocols.
	Protocol string

	// Optional - enables or disables Half-Closed support for the load balancer.
	// Half-Closed support provides the ability for one end of the connection to
	// terminate its output, while still receiving data from the other end. Only
	// available for TCP/TCP_CLIENT_FIRST protocols.
	HalfClosed gophercloud.EnabledState

	// Optional - the type of virtual IPs you want associated with the load
	// balancer.
	VIPs []vips.VIP

	// Optional - the access list management feature allows fine-grained network
	// access controls to be applied to the load balancer virtual IP address.
	AccessList *acl.AccessList

	// Optional - algorithm that defines how traffic should be directed between
	// back-end nodes.
	Algorithm string

	// Optional - current connection logging configuration.
	ConnectionLogging *ConnectionLogging

	// Optional - specifies a limit on the number of connections per IP address
	// to help mitigate malicious or abusive traffic to your applications.
	ConnThrottle *throttle.ConnectionThrottle

	// Optional - the type of health monitor check to perform to ensure that the
	// service is performing properly.
	HealthMonitor *monitors.Monitor

	// Optional - arbitrary information that can be associated with each LB.
	Metadata map[string]interface{}

	// Optional - port number for the service you are load balancing.
	Port int

	// Optional - the timeout value for the load balancer and communications with
	// its nodes. Defaults to 30 seconds with a maximum of 120 seconds.
	Timeout int

	// Optional - specifies whether multiple requests from clients are directed
	// to the same node.
	SessionPersistence *sessions.SessionPersistence

	// Optional - enables or disables HTTP to HTTPS redirection for the load
	// balancer. When enabled, any HTTP request returns status code 301 (Moved
	// Permanently), and the requester is redirected to the requested URL via the
	// HTTPS protocol on port 443. For example, http://example.com/page.html
	// would be redirected to https://example.com/page.html. Only available for
	// HTTPS protocol (port=443), or HTTP protocol with a properly configured SSL
	// termination (secureTrafficOnly=true, securePort=443).
	HTTPSRedirect gophercloud.EnabledState
}

// ToLBCreateMap casts a CreateOpts struct to a map.
func (opts CreateOpts) ToLBCreateMap() (map[string]interface{}, error) {
	lb := make(map[string]interface{})

	if opts.Name == "" {
		return lb, errNameRequired
	}
	if opts.Timeout > 120 {
		return lb, errTimeoutExceeded
	}

	lb["name"] = opts.Name

	if len(opts.Nodes) > 0 {
		nodes := []map[string]interface{}{}
		for _, n := range opts.Nodes {
			nodes = append(nodes, map[string]interface{}{
				"address":   n.Address,
				"port":      n.Port,
				"condition": n.Condition,
			})
		}
		lb["nodes"] = nodes
	}

	if opts.Protocol != "" {
		lb["protocol"] = opts.Protocol
	}
	if opts.HalfClosed != nil {
		lb["halfClosed"] = opts.HalfClosed
	}
	if len(opts.VIPs) > 0 {
		lb["virtualIps"] = opts.VIPs
	}
	if opts.AccessList != nil {
		lb["accessList"] = &opts.AccessList
	}
	if opts.Algorithm != "" {
		lb["algorithm"] = opts.Algorithm
	}
	if opts.ConnectionLogging != nil {
		lb["connectionLogging"] = &opts.ConnectionLogging
	}
	if opts.ConnThrottle != nil {
		lb["connectionThrottle"] = &opts.ConnThrottle
	}
	if opts.HealthMonitor != nil {
		lb["healthMonitor"] = &opts.HealthMonitor
	}
	if len(opts.Metadata) != 0 {
		lb["metadata"] = opts.Metadata
	}
	if opts.Port > 0 {
		lb["port"] = opts.Port
	}
	if opts.Timeout > 0 {
		lb["timeout"] = opts.Timeout
	}
	if opts.SessionPersistence != nil {
		lb["sessionPersistence"] = &opts.SessionPersistence
	}
	if opts.HTTPSRedirect != nil {
		lb["httpsRedirect"] = &opts.HTTPSRedirect
	}

	return map[string]interface{}{"loadBalancer": lb}, nil
}

// Create is the operation responsible for asynchronously provisioning a new
// load balancer based on the configuration defined in CreateOpts. Once the
// request is validated and progress has started on the provisioning process, a
// response struct is returned. When extracted (with Extract()), you have
// to the load balancer's unique ID and status.
//
// Once an ID is attained, you can check on the progress of the operation by
// calling Get and passing in the ID. If the corresponding request cannot be
// fulfilled due to insufficient or invalid data, an HTTP 400 (Bad Request)
// error response is returned with information regarding the nature of the
// failure in the body of the response. Failures in the validation process are
// non-recoverable and require the caller to correct the cause of the failure.
func Create(c *gophercloud.ServiceClient, opts CreateOptsBuilder) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToLBCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = c.Post(rootURL(c), reqBody, &res.Body, nil)
	return res
}

// Get is the operation responsible for providing detailed information
// regarding a specific load balancer which is configured and associated with
// your account. This operation is not capable of returning details for a load
// balancer which has been deleted.
func Get(c *gophercloud.ServiceClient, id int) GetResult {
	var res GetResult

	_, res.Err = c.Get(resourceURL(c, id), &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})

	return res
}

// BulkDelete removes all the load balancers referenced in the slice of IDs.
// Any and all configuration data associated with these load balancers is
// immediately purged and is not recoverable.
//
// If one of the items in the list cannot be removed due to its current status,
// a 400 Bad Request error is returned along with the IDs of the ones the
// system identified as potential failures for this request.
func BulkDelete(c *gophercloud.ServiceClient, ids []int) DeleteResult {
	var res DeleteResult

	if len(ids) > 10 || len(ids) == 0 {
		res.Err = errors.New("You must provide a minimum of 1 and a maximum of 10 LB IDs")
		return res
	}

	url := rootURL(c)
	url += gophercloud.IDSliceToQueryString("id", ids)

	_, res.Err = c.Delete(url, nil)
	return res
}

// Delete removes a single load balancer.
func Delete(c *gophercloud.ServiceClient, id int) DeleteResult {
	var res DeleteResult
	_, res.Err = c.Delete(resourceURL(c, id), nil)
	return res
}

// UpdateOptsBuilder represents a type that can be converted into a JSON-like
// map structure.
type UpdateOptsBuilder interface {
	ToLBUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts represents the options for updating an existing load balancer.
type UpdateOpts struct {
	// Optional - new name of the load balancer.
	Name string

	// Optional - the new protocol you want your load balancer to have.
	// See http://docs.rackspace.com/loadbalancers/api/v1.0/clb-devguide/content/protocols.html
	// for a full list of supported protocols.
	Protocol string

	// Optional - see the HalfClosed field in CreateOpts for more information.
	HalfClosed gophercloud.EnabledState

	// Optional - see the Algorithm field in CreateOpts for more information.
	Algorithm string

	// Optional - see the Port field in CreateOpts for more information.
	Port int

	// Optional - see the Timeout field in CreateOpts for more information.
	Timeout int

	// Optional - see the HTTPSRedirect field in CreateOpts for more information.
	HTTPSRedirect gophercloud.EnabledState
}

// ToLBUpdateMap casts an UpdateOpts struct to a map.
func (opts UpdateOpts) ToLBUpdateMap() (map[string]interface{}, error) {
	lb := make(map[string]interface{})

	if opts.Name != "" {
		lb["name"] = opts.Name
	}
	if opts.Protocol != "" {
		lb["protocol"] = opts.Protocol
	}
	if opts.HalfClosed != nil {
		lb["halfClosed"] = opts.HalfClosed
	}
	if opts.Algorithm != "" {
		lb["algorithm"] = opts.Algorithm
	}
	if opts.Port > 0 {
		lb["port"] = opts.Port
	}
	if opts.Timeout > 0 {
		lb["timeout"] = opts.Timeout
	}
	if opts.HTTPSRedirect != nil {
		lb["httpsRedirect"] = &opts.HTTPSRedirect
	}

	return map[string]interface{}{"loadBalancer": lb}, nil
}

// Update is the operation responsible for asynchronously updating the
// attributes of a specific load balancer. Upon successful validation of the
// request, the service returns a 202 Accepted response, and the load balancer
// enters a PENDING_UPDATE state. A user can poll the load balancer with Get to
// wait for the changes to be applied. When this happens, the load balancer will
// return to an ACTIVE state.
func Update(c *gophercloud.ServiceClient, id int, opts UpdateOptsBuilder) UpdateResult {
	var res UpdateResult

	reqBody, err := opts.ToLBUpdateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = c.Put(resourceURL(c, id), reqBody, nil, nil)
	return res
}

// ListProtocols is the operation responsible for returning a paginated
// collection of load balancer protocols.
func ListProtocols(client *gophercloud.ServiceClient) pagination.Pager {
	url := protocolsURL(client)
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return ProtocolPage{pagination.SinglePageBase(r)}
	})
}

// ListAlgorithms is the operation responsible for returning a paginated
// collection of load balancer algorithms.
func ListAlgorithms(client *gophercloud.ServiceClient) pagination.Pager {
	url := algorithmsURL(client)
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return AlgorithmPage{pagination.SinglePageBase(r)}
	})
}

// IsLoggingEnabled returns true if the load balancer has connection logging
// enabled and false if not.
func IsLoggingEnabled(client *gophercloud.ServiceClient, id int) (bool, error) {
	var body interface{}

	_, err := client.Get(loggingURL(client, id), &body, nil)
	if err != nil {
		return false, err
	}

	var resp struct {
		CL struct {
			Enabled bool `mapstructure:"enabled"`
		} `mapstructure:"connectionLogging"`
	}

	err = mapstructure.Decode(body, &resp)
	return resp.CL.Enabled, err
}

func toConnLoggingMap(state bool) map[string]map[string]bool {
	return map[string]map[string]bool{
		"connectionLogging": map[string]bool{"enabled": state},
	}
}

// EnableLogging will enable connection logging for a specified load balancer.
func EnableLogging(client *gophercloud.ServiceClient, id int) gophercloud.ErrResult {
	var res gophercloud.ErrResult
	_, res.Err = client.Put(loggingURL(client, id), toConnLoggingMap(true), nil, nil)
	return res
}

// DisableLogging will disable connection logging for a specified load balancer.
func DisableLogging(client *gophercloud.ServiceClient, id int) gophercloud.ErrResult {
	var res gophercloud.ErrResult
	_, res.Err = client.Put(loggingURL(client, id), toConnLoggingMap(false), nil, nil)
	return res
}

// GetErrorPage will retrieve the current error page for the load balancer.
func GetErrorPage(client *gophercloud.ServiceClient, id int) ErrorPageResult {
	var res ErrorPageResult
	_, res.Err = client.Get(errorPageURL(client, id), &res.Body, nil)
	return res
}

// SetErrorPage will set the HTML of the load balancer's error page to a
// specific value.
func SetErrorPage(client *gophercloud.ServiceClient, id int, html string) ErrorPageResult {
	var res ErrorPageResult

	type stringMap map[string]string
	reqBody := map[string]stringMap{"errorpage": stringMap{"content": html}}

	_, res.Err = client.Put(errorPageURL(client, id), reqBody, &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})

	return res
}

// DeleteErrorPage will delete the current error page for the load balancer.
func DeleteErrorPage(client *gophercloud.ServiceClient, id int) gophercloud.ErrResult {
	var res gophercloud.ErrResult
	_, res.Err = client.Delete(errorPageURL(client, id), &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return res
}

// GetStats will retrieve detailed stats related to the load balancer's usage.
func GetStats(client *gophercloud.ServiceClient, id int) StatsResult {
	var res StatsResult
	_, res.Err = client.Get(statsURL(client, id), &res.Body, nil)
	return res
}

// IsContentCached will check to see whether the specified load balancer caches
// content. When content caching is enabled, recently-accessed files are stored
// on the load balancer for easy retrieval by web clients. Content caching
// improves the performance of high traffic web sites by temporarily storing
// data that was recently accessed. While it's cached, requests for that data
// are served by the load balancer, which in turn reduces load off the back-end
// nodes. The result is improved response times for those requests and less
// load on the web server.
func IsContentCached(client *gophercloud.ServiceClient, id int) (bool, error) {
	var body interface{}

	_, err := client.Get(cacheURL(client, id), &body, nil)
	if err != nil {
		return false, err
	}

	var resp struct {
		CC struct {
			Enabled bool `mapstructure:"enabled"`
		} `mapstructure:"contentCaching"`
	}

	err = mapstructure.Decode(body, &resp)
	return resp.CC.Enabled, err
}

func toCachingMap(state bool) map[string]map[string]bool {
	return map[string]map[string]bool{
		"contentCaching": map[string]bool{"enabled": state},
	}
}

// EnableCaching will enable content-caching for the specified load balancer.
func EnableCaching(client *gophercloud.ServiceClient, id int) gophercloud.ErrResult {
	var res gophercloud.ErrResult
	_, res.Err = client.Put(cacheURL(client, id), toCachingMap(true), nil, nil)
	return res
}

// DisableCaching will disable content-caching for the specified load balancer.
func DisableCaching(client *gophercloud.ServiceClient, id int) gophercloud.ErrResult {
	var res gophercloud.ErrResult
	_, res.Err = client.Put(cacheURL(client, id), toCachingMap(false), nil, nil)
	return res
}
