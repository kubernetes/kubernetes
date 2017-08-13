package godo

import (
	"fmt"

	"github.com/digitalocean/godo/context"
)

const loadBalancersBasePath = "/v2/load_balancers"
const forwardingRulesPath = "forwarding_rules"
const dropletsPath = "droplets"

// LoadBalancersService is an interface for managing load balancers with the DigitalOcean API.
// See: https://developers.digitalocean.com/documentation/v2#load-balancers
type LoadBalancersService interface {
	Get(context.Context, string) (*LoadBalancer, *Response, error)
	List(context.Context, *ListOptions) ([]LoadBalancer, *Response, error)
	Create(context.Context, *LoadBalancerRequest) (*LoadBalancer, *Response, error)
	Update(ctx context.Context, lbID string, lbr *LoadBalancerRequest) (*LoadBalancer, *Response, error)
	Delete(ctx context.Context, lbID string) (*Response, error)
	AddDroplets(ctx context.Context, lbID string, dropletIDs ...int) (*Response, error)
	RemoveDroplets(ctx context.Context, lbID string, dropletIDs ...int) (*Response, error)
	AddForwardingRules(ctx context.Context, lbID string, rules ...ForwardingRule) (*Response, error)
	RemoveForwardingRules(ctx context.Context, lbID string, rules ...ForwardingRule) (*Response, error)
}

// LoadBalancer represents a DigitalOcean load balancer configuration.
type LoadBalancer struct {
	ID                  string           `json:"id,omitempty"`
	Name                string           `json:"name,omitempty"`
	IP                  string           `json:"ip,omitempty"`
	Algorithm           string           `json:"algorithm,omitempty"`
	Status              string           `json:"status,omitempty"`
	Created             string           `json:"created_at,omitempty"`
	ForwardingRules     []ForwardingRule `json:"forwarding_rules,omitempty"`
	HealthCheck         *HealthCheck     `json:"health_check,omitempty"`
	StickySessions      *StickySessions  `json:"sticky_sessions,omitempty"`
	Region              *Region          `json:"region,omitempty"`
	DropletIDs          []int            `json:"droplet_ids,omitempty"`
	Tag                 string           `json:"tag,omitempty"`
	RedirectHttpToHttps bool             `json:"redirect_http_to_https,omitempty"`
}

// String creates a human-readable description of a LoadBalancer.
func (l LoadBalancer) String() string {
	return Stringify(l)
}

// ForwardingRule represents load balancer forwarding rules.
type ForwardingRule struct {
	EntryProtocol  string `json:"entry_protocol,omitempty"`
	EntryPort      int    `json:"entry_port,omitempty"`
	TargetProtocol string `json:"target_protocol,omitempty"`
	TargetPort     int    `json:"target_port,omitempty"`
	CertificateID  string `json:"certificate_id,omitempty"`
	TlsPassthrough bool   `json:"tls_passthrough,omitempty"`
}

// String creates a human-readable description of a ForwardingRule.
func (f ForwardingRule) String() string {
	return Stringify(f)
}

// HealthCheck represents optional load balancer health check rules.
type HealthCheck struct {
	Protocol               string `json:"protocol,omitempty"`
	Port                   int    `json:"port,omitempty"`
	Path                   string `json:"path,omitempty"`
	CheckIntervalSeconds   int    `json:"check_interval_seconds,omitempty"`
	ResponseTimeoutSeconds int    `json:"response_timeout_seconds,omitempty"`
	HealthyThreshold       int    `json:"healthy_threshold,omitempty"`
	UnhealthyThreshold     int    `json:"unhealthy_threshold,omitempty"`
}

// String creates a human-readable description of a HealthCheck.
func (h HealthCheck) String() string {
	return Stringify(h)
}

// StickySessions represents optional load balancer session affinity rules.
type StickySessions struct {
	Type             string `json:"type,omitempty"`
	CookieName       string `json:"cookie_name,omitempty"`
	CookieTtlSeconds int    `json:"cookie_ttl_seconds,omitempty"`
}

// String creates a human-readable description of a StickySessions instance.
func (s StickySessions) String() string {
	return Stringify(s)
}

// LoadBalancerRequest represents the configuration to be applied to an existing or a new load balancer.
type LoadBalancerRequest struct {
	Name                string           `json:"name,omitempty"`
	Algorithm           string           `json:"algorithm,omitempty"`
	Region              string           `json:"region,omitempty"`
	ForwardingRules     []ForwardingRule `json:"forwarding_rules,omitempty"`
	HealthCheck         *HealthCheck     `json:"health_check,omitempty"`
	StickySessions      *StickySessions  `json:"sticky_sessions,omitempty"`
	DropletIDs          []int            `json:"droplet_ids,omitempty"`
	Tag                 string           `json:"tag,omitempty"`
	RedirectHttpToHttps bool             `json:"redirect_http_to_https,omitempty"`
}

// String creates a human-readable description of a LoadBalancerRequest.
func (l LoadBalancerRequest) String() string {
	return Stringify(l)
}

type forwardingRulesRequest struct {
	Rules []ForwardingRule `json:"forwarding_rules,omitempty"`
}

func (l forwardingRulesRequest) String() string {
	return Stringify(l)
}

type dropletIDsRequest struct {
	IDs []int `json:"droplet_ids,omitempty"`
}

func (l dropletIDsRequest) String() string {
	return Stringify(l)
}

type loadBalancersRoot struct {
	LoadBalancers []LoadBalancer `json:"load_balancers"`
	Links         *Links         `json:"links"`
}

type loadBalancerRoot struct {
	LoadBalancer *LoadBalancer `json:"load_balancer"`
}

// LoadBalancersServiceOp handles communication with load balancer-related methods of the DigitalOcean API.
type LoadBalancersServiceOp struct {
	client *Client
}

var _ LoadBalancersService = &LoadBalancersServiceOp{}

// Get an existing load balancer by its identifier.
func (l *LoadBalancersServiceOp) Get(ctx context.Context, lbID string) (*LoadBalancer, *Response, error) {
	path := fmt.Sprintf("%s/%s", loadBalancersBasePath, lbID)

	req, err := l.client.NewRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(loadBalancerRoot)
	resp, err := l.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}

	return root.LoadBalancer, resp, err
}

// List load balancers, with optional pagination.
func (l *LoadBalancersServiceOp) List(ctx context.Context, opt *ListOptions) ([]LoadBalancer, *Response, error) {
	path, err := addOptions(loadBalancersBasePath, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := l.client.NewRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(loadBalancersRoot)
	resp, err := l.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}
	if l := root.Links; l != nil {
		resp.Links = l
	}

	return root.LoadBalancers, resp, err
}

// Create a new load balancer with a given configuration.
func (l *LoadBalancersServiceOp) Create(ctx context.Context, lbr *LoadBalancerRequest) (*LoadBalancer, *Response, error) {
	req, err := l.client.NewRequest(ctx, "POST", loadBalancersBasePath, lbr)
	if err != nil {
		return nil, nil, err
	}

	root := new(loadBalancerRoot)
	resp, err := l.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}

	return root.LoadBalancer, resp, err
}

// Update an existing load balancer with new configuration.
func (l *LoadBalancersServiceOp) Update(ctx context.Context, lbID string, lbr *LoadBalancerRequest) (*LoadBalancer, *Response, error) {
	path := fmt.Sprintf("%s/%s", loadBalancersBasePath, lbID)

	req, err := l.client.NewRequest(ctx, "PUT", path, lbr)
	if err != nil {
		return nil, nil, err
	}

	root := new(loadBalancerRoot)
	resp, err := l.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}

	return root.LoadBalancer, resp, err
}

// Delete a load balancer by its identifier.
func (l *LoadBalancersServiceOp) Delete(ctx context.Context, ldID string) (*Response, error) {
	path := fmt.Sprintf("%s/%s", loadBalancersBasePath, ldID)

	req, err := l.client.NewRequest(ctx, "DELETE", path, nil)
	if err != nil {
		return nil, err
	}

	return l.client.Do(ctx, req, nil)
}

// AddDroplets adds droplets to a load balancer.
func (l *LoadBalancersServiceOp) AddDroplets(ctx context.Context, lbID string, dropletIDs ...int) (*Response, error) {
	path := fmt.Sprintf("%s/%s/%s", loadBalancersBasePath, lbID, dropletsPath)

	req, err := l.client.NewRequest(ctx, "POST", path, &dropletIDsRequest{IDs: dropletIDs})
	if err != nil {
		return nil, err
	}

	return l.client.Do(ctx, req, nil)
}

// RemoveDroplets removes droplets from a load balancer.
func (l *LoadBalancersServiceOp) RemoveDroplets(ctx context.Context, lbID string, dropletIDs ...int) (*Response, error) {
	path := fmt.Sprintf("%s/%s/%s", loadBalancersBasePath, lbID, dropletsPath)

	req, err := l.client.NewRequest(ctx, "DELETE", path, &dropletIDsRequest{IDs: dropletIDs})
	if err != nil {
		return nil, err
	}

	return l.client.Do(ctx, req, nil)
}

// AddForwardingRules adds forwarding rules to a load balancer.
func (l *LoadBalancersServiceOp) AddForwardingRules(ctx context.Context, lbID string, rules ...ForwardingRule) (*Response, error) {
	path := fmt.Sprintf("%s/%s/%s", loadBalancersBasePath, lbID, forwardingRulesPath)

	req, err := l.client.NewRequest(ctx, "POST", path, &forwardingRulesRequest{Rules: rules})
	if err != nil {
		return nil, err
	}

	return l.client.Do(ctx, req, nil)
}

// RemoveForwardingRules removes forwarding rules from a load balancer.
func (l *LoadBalancersServiceOp) RemoveForwardingRules(ctx context.Context, lbID string, rules ...ForwardingRule) (*Response, error) {
	path := fmt.Sprintf("%s/%s/%s", loadBalancersBasePath, lbID, forwardingRulesPath)

	req, err := l.client.NewRequest(ctx, "DELETE", path, &forwardingRulesRequest{Rules: rules})
	if err != nil {
		return nil, err
	}

	return l.client.Do(ctx, req, nil)
}
