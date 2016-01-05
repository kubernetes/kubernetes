package lbs

import (
	"reflect"
	"time"

	"github.com/mitchellh/mapstructure"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/lb/v1/acl"
	"github.com/rackspace/gophercloud/rackspace/lb/v1/nodes"
	"github.com/rackspace/gophercloud/rackspace/lb/v1/sessions"
	"github.com/rackspace/gophercloud/rackspace/lb/v1/throttle"
	"github.com/rackspace/gophercloud/rackspace/lb/v1/vips"
)

// Protocol represents the network protocol which the load balancer accepts.
type Protocol struct {
	// The name of the protocol, e.g. HTTP, LDAP, FTP, etc.
	Name string

	// The port number for the protocol.
	Port int
}

// Algorithm defines how traffic should be directed between back-end nodes.
type Algorithm struct {
	// The name of the algorithm, e.g RANDOM, ROUND_ROBIN, etc.
	Name string
}

// Status represents the potential state of a load balancer resource.
type Status string

const (
	// ACTIVE indicates that the LB is configured properly and ready to serve
	// traffic to incoming requests via the configured virtual IPs.
	ACTIVE Status = "ACTIVE"

	// BUILD indicates that the LB is being provisioned for the first time and
	// configuration is being applied to bring the service online. The service
	// cannot yet serve incoming requests.
	BUILD Status = "BUILD"

	// PENDINGUPDATE indicates that the LB is online but configuration changes
	// are being applied to update the service based on a previous request.
	PENDINGUPDATE Status = "PENDING_UPDATE"

	// PENDINGDELETE indicates that the LB is online but configuration changes
	// are being applied to begin deletion of the service based on a previous
	// request.
	PENDINGDELETE Status = "PENDING_DELETE"

	// SUSPENDED indicates that the LB has been taken offline and disabled.
	SUSPENDED Status = "SUSPENDED"

	// ERROR indicates that the system encountered an error when attempting to
	// configure the load balancer.
	ERROR Status = "ERROR"

	// DELETED indicates that the LB has been deleted.
	DELETED Status = "DELETED"
)

// Datetime represents the structure of a Created or Updated field.
type Datetime struct {
	Time time.Time `mapstructure:"-"`
}

// LoadBalancer represents a load balancer API resource.
type LoadBalancer struct {
	// Human-readable name for the load balancer.
	Name string

	// The unique ID for the load balancer.
	ID int

	// Represents the service protocol being load balanced. See Protocol type for
	// a list of accepted values.
	// See http://docs.rackspace.com/loadbalancers/api/v1.0/clb-devguide/content/protocols.html
	// for a full list of supported protocols.
	Protocol string

	// Defines how traffic should be directed between back-end nodes. The default
	// algorithm is RANDOM. See Algorithm type for a list of accepted values.
	Algorithm string

	// The current status of the load balancer.
	Status Status

	// The number of load balancer nodes.
	NodeCount int `mapstructure:"nodeCount"`

	// Slice of virtual IPs associated with this load balancer.
	VIPs []vips.VIP `mapstructure:"virtualIps"`

	// Datetime when the LB was created.
	Created Datetime

	// Datetime when the LB was created.
	Updated Datetime

	// Port number for the service you are load balancing.
	Port int

	// HalfClosed provides the ability for one end of the connection to
	// terminate its output while still receiving data from the other end. This
	// is only available on TCP/TCP_CLIENT_FIRST protocols.
	HalfClosed bool

	// Timeout represents the timeout value between a load balancer and its
	// nodes. Defaults to 30 seconds with a maximum of 120 seconds.
	Timeout int

	// The cluster name.
	Cluster Cluster

	// Nodes shows all the back-end nodes which are associated with the load
	// balancer. These are the devices which are delivered traffic.
	Nodes []nodes.Node

	// Current connection logging configuration.
	ConnectionLogging ConnectionLogging

	// SessionPersistence specifies whether multiple requests from clients are
	// directed to the same node.
	SessionPersistence sessions.SessionPersistence

	// ConnectionThrottle specifies a limit on the number of connections per IP
	// address to help mitigate malicious or abusive traffic to your applications.
	ConnectionThrottle throttle.ConnectionThrottle

	// The source public and private IP addresses.
	SourceAddrs SourceAddrs `mapstructure:"sourceAddresses"`

	// Represents the access rules for this particular load balancer. IP addresses
	// or subnet ranges, depending on their type (ALLOW or DENY), can be permitted
	// or blocked.
	AccessList acl.AccessList
}

// SourceAddrs represents the source public and private IP addresses.
type SourceAddrs struct {
	IPv4Public  string `json:"ipv4Public" mapstructure:"ipv4Public"`
	IPv4Private string `json:"ipv4Servicenet" mapstructure:"ipv4Servicenet"`
	IPv6Public  string `json:"ipv6Public" mapstructure:"ipv6Public"`
	IPv6Private string `json:"ipv6Servicenet" mapstructure:"ipv6Servicenet"`
}

// ConnectionLogging - temp
type ConnectionLogging struct {
	Enabled bool
}

// Cluster - temp
type Cluster struct {
	Name string
}

// LBPage is the page returned by a pager when traversing over a collection of
// LBs.
type LBPage struct {
	pagination.LinkedPageBase
}

// IsEmpty checks whether a NetworkPage struct is empty.
func (p LBPage) IsEmpty() (bool, error) {
	is, err := ExtractLBs(p)
	if err != nil {
		return true, nil
	}
	return len(is) == 0, nil
}

// ExtractLBs accepts a Page struct, specifically a LBPage struct, and extracts
// the elements into a slice of LoadBalancer structs. In other words, a generic
// collection is mapped into a relevant slice.
func ExtractLBs(page pagination.Page) ([]LoadBalancer, error) {
	var resp struct {
		LBs []LoadBalancer `mapstructure:"loadBalancers" json:"loadBalancers"`
	}

	coll := page.(LBPage).Body
	err := mapstructure.Decode(coll, &resp)

	s := reflect.ValueOf(coll.(map[string]interface{})["loadBalancers"])

	for i := 0; i < s.Len(); i++ {
		val := (s.Index(i).Interface()).(map[string]interface{})

		ts, err := extractTS(val, "created")
		if err != nil {
			return resp.LBs, err
		}
		resp.LBs[i].Created.Time = ts

		ts, err = extractTS(val, "updated")
		if err != nil {
			return resp.LBs, err
		}
		resp.LBs[i].Updated.Time = ts
	}

	return resp.LBs, err
}

func extractTS(body map[string]interface{}, key string) (time.Time, error) {
	val := body[key].(map[string]interface{})
	return time.Parse(time.RFC3339, val["time"].(string))
}

type commonResult struct {
	gophercloud.Result
}

// Extract interprets any commonResult as a LB, if possible.
func (r commonResult) Extract() (*LoadBalancer, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var response struct {
		LB LoadBalancer `mapstructure:"loadBalancer"`
	}

	err := mapstructure.Decode(r.Body, &response)

	json := r.Body.(map[string]interface{})
	lb := json["loadBalancer"].(map[string]interface{})

	ts, err := extractTS(lb, "created")
	if err != nil {
		return nil, err
	}
	response.LB.Created.Time = ts

	ts, err = extractTS(lb, "updated")
	if err != nil {
		return nil, err
	}
	response.LB.Updated.Time = ts

	return &response.LB, err
}

// CreateResult represents the result of a create operation.
type CreateResult struct {
	commonResult
}

// DeleteResult represents the result of a delete operation.
type DeleteResult struct {
	gophercloud.ErrResult
}

// UpdateResult represents the result of an update operation.
type UpdateResult struct {
	gophercloud.ErrResult
}

// GetResult represents the result of a get operation.
type GetResult struct {
	commonResult
}

// ProtocolPage is the page returned by a pager when traversing over a
// collection of LB protocols.
type ProtocolPage struct {
	pagination.SinglePageBase
}

// IsEmpty checks whether a ProtocolPage struct is empty.
func (p ProtocolPage) IsEmpty() (bool, error) {
	is, err := ExtractProtocols(p)
	if err != nil {
		return true, nil
	}
	return len(is) == 0, nil
}

// ExtractProtocols accepts a Page struct, specifically a ProtocolPage struct,
// and extracts the elements into a slice of Protocol structs. In other
// words, a generic collection is mapped into a relevant slice.
func ExtractProtocols(page pagination.Page) ([]Protocol, error) {
	var resp struct {
		Protocols []Protocol `mapstructure:"protocols" json:"protocols"`
	}
	err := mapstructure.Decode(page.(ProtocolPage).Body, &resp)
	return resp.Protocols, err
}

// AlgorithmPage is the page returned by a pager when traversing over a
// collection of LB algorithms.
type AlgorithmPage struct {
	pagination.SinglePageBase
}

// IsEmpty checks whether an AlgorithmPage struct is empty.
func (p AlgorithmPage) IsEmpty() (bool, error) {
	is, err := ExtractAlgorithms(p)
	if err != nil {
		return true, nil
	}
	return len(is) == 0, nil
}

// ExtractAlgorithms accepts a Page struct, specifically an AlgorithmPage struct,
// and extracts the elements into a slice of Algorithm structs. In other
// words, a generic collection is mapped into a relevant slice.
func ExtractAlgorithms(page pagination.Page) ([]Algorithm, error) {
	var resp struct {
		Algorithms []Algorithm `mapstructure:"algorithms" json:"algorithms"`
	}
	err := mapstructure.Decode(page.(AlgorithmPage).Body, &resp)
	return resp.Algorithms, err
}

// ErrorPage represents the HTML file that is shown to an end user who is
// attempting to access a load balancer node that is offline/unavailable.
//
// During provisioning, every load balancer is configured with a default error
// page that gets displayed when traffic is requested for an offline node.
//
// You can add a single custom error page with an HTTP-based protocol to a load
// balancer. Page updates override existing content. If a custom error page is
// deleted, or the load balancer is changed to a non-HTTP protocol, the default
// error page is restored.
type ErrorPage struct {
	Content string
}

// ErrorPageResult represents the result of an error page operation -
// specifically getting or creating one.
type ErrorPageResult struct {
	gophercloud.Result
}

// Extract interprets any commonResult as an ErrorPage, if possible.
func (r ErrorPageResult) Extract() (*ErrorPage, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var response struct {
		ErrorPage ErrorPage `mapstructure:"errorpage"`
	}

	err := mapstructure.Decode(r.Body, &response)

	return &response.ErrorPage, err
}

// Stats represents all the key information about a load balancer's usage.
type Stats struct {
	// The number of connections closed by this load balancer because its
	// ConnectTimeout interval was exceeded.
	ConnectTimeout int `mapstructure:"connectTimeOut"`

	// The number of transaction or protocol errors for this load balancer.
	ConnectError int

	// Number of connection failures for this load balancer.
	ConnectFailure int

	// Number of connections closed by this load balancer because its Timeout
	// interval was exceeded.
	DataTimedOut int

	// Number of connections closed by this load balancer because the
	// 'keepalive_timeout' interval was exceeded.
	KeepAliveTimedOut int

	// The maximum number of simultaneous TCP connections this load balancer has
	// processed at any one time.
	MaxConnections int `mapstructure:"maxConn"`

	// Number of simultaneous connections active at the time of the request.
	CurrentConnections int `mapstructure:"currentConn"`

	// Number of SSL connections closed by this load balancer because the
	// ConnectTimeout interval was exceeded.
	SSLConnectTimeout int `mapstructure:"connectTimeOutSsl"`

	// Number of SSL transaction or protocol erros in this load balancer.
	SSLConnectError int `mapstructure:"connectErrorSsl"`

	// Number of SSL connection failures in this load balancer.
	SSLConnectFailure int `mapstructure:"connectFailureSsl"`

	// Number of SSL connections closed by this load balancer because the
	// Timeout interval was exceeded.
	SSLDataTimedOut int `mapstructure:"dataTimedOutSsl"`

	// Number of SSL connections closed by this load balancer because the
	// 'keepalive_timeout' interval was exceeded.
	SSLKeepAliveTimedOut int `mapstructure:"keepAliveTimedOutSsl"`

	// Maximum number of simultaneous SSL connections this load balancer has
	// processed at any one time.
	SSLMaxConnections int `mapstructure:"maxConnSsl"`

	// Number of simultaneous SSL connections active at the time of the request.
	SSLCurrentConnections int `mapstructure:"currentConnSsl"`
}

// StatsResult represents the result of a Stats operation.
type StatsResult struct {
	gophercloud.Result
}

// Extract interprets any commonResult as a Stats struct, if possible.
func (r StatsResult) Extract() (*Stats, error) {
	if r.Err != nil {
		return nil, r.Err
	}
	res := &Stats{}
	err := mapstructure.Decode(r.Body, res)
	return res, err
}
