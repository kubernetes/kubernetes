package godo

import (
	"path"
	"strconv"

	"github.com/digitalocean/godo/context"
)

const firewallsBasePath = "/v2/firewalls"

// FirewallsService is an interface for managing Firewalls with the DigitalOcean API.
// See: https://developers.digitalocean.com/documentation/documentation/v2/#firewalls
type FirewallsService interface {
	Get(context.Context, string) (*Firewall, *Response, error)
	Create(context.Context, *FirewallRequest) (*Firewall, *Response, error)
	Update(context.Context, string, *FirewallRequest) (*Firewall, *Response, error)
	Delete(context.Context, string) (*Response, error)
	List(context.Context, *ListOptions) ([]Firewall, *Response, error)
	ListByDroplet(context.Context, int, *ListOptions) ([]Firewall, *Response, error)
	AddDroplets(context.Context, string, ...int) (*Response, error)
	RemoveDroplets(context.Context, string, ...int) (*Response, error)
	AddTags(context.Context, string, ...string) (*Response, error)
	RemoveTags(context.Context, string, ...string) (*Response, error)
	AddRules(context.Context, string, *FirewallRulesRequest) (*Response, error)
	RemoveRules(context.Context, string, *FirewallRulesRequest) (*Response, error)
}

// FirewallsServiceOp handles communication with Firewalls methods of the DigitalOcean API.
type FirewallsServiceOp struct {
	client *Client
}

// Firewall represents a DigitalOcean Firewall configuration.
type Firewall struct {
	ID             string          `json:"id"`
	Name           string          `json:"name"`
	Status         string          `json:"status"`
	InboundRules   []InboundRule   `json:"inbound_rules"`
	OutboundRules  []OutboundRule  `json:"outbound_rules"`
	DropletIDs     []int           `json:"droplet_ids"`
	Tags           []string        `json:"tags"`
	Created        string          `json:"created_at"`
	PendingChanges []PendingChange `json:"pending_changes"`
}

// String creates a human-readable description of a Firewall.
func (fw Firewall) String() string {
	return Stringify(fw)
}

// FirewallRequest represents the configuration to be applied to an existing or a new Firewall.
type FirewallRequest struct {
	Name          string         `json:"name"`
	InboundRules  []InboundRule  `json:"inbound_rules"`
	OutboundRules []OutboundRule `json:"outbound_rules"`
	DropletIDs    []int          `json:"droplet_ids"`
	Tags          []string       `json:"tags"`
}

// FirewallRulesRequest represents rules configuration to be applied to an existing Firewall.
type FirewallRulesRequest struct {
	InboundRules  []InboundRule  `json:"inbound_rules"`
	OutboundRules []OutboundRule `json:"outbound_rules"`
}

// InboundRule represents a DigitalOcean Firewall inbound rule.
type InboundRule struct {
	Protocol  string   `json:"protocol,omitempty"`
	PortRange string   `json:"ports,omitempty"`
	Sources   *Sources `json:"sources"`
}

// OutboundRule represents a DigitalOcean Firewall outbound rule.
type OutboundRule struct {
	Protocol     string        `json:"protocol,omitempty"`
	PortRange    string        `json:"ports,omitempty"`
	Destinations *Destinations `json:"destinations"`
}

// Sources represents a DigitalOcean Firewall InboundRule sources.
type Sources struct {
	Addresses        []string `json:"addresses,omitempty"`
	Tags             []string `json:"tags,omitempty"`
	DropletIDs       []int    `json:"droplet_ids,omitempty"`
	LoadBalancerUIDs []string `json:"load_balancer_uids,omitempty"`
}

// PendingChange represents a DigitalOcean Firewall status details.
type PendingChange struct {
	DropletID int    `json:"droplet_id,omitempty"`
	Removing  bool   `json:"removing,omitempty"`
	Status    string `json:"status,omitempty"`
}

// Destinations represents a DigitalOcean Firewall OutboundRule destinations.
type Destinations struct {
	Addresses        []string `json:"addresses,omitempty"`
	Tags             []string `json:"tags,omitempty"`
	DropletIDs       []int    `json:"droplet_ids,omitempty"`
	LoadBalancerUIDs []string `json:"load_balancer_uids,omitempty"`
}

var _ FirewallsService = &FirewallsServiceOp{}

// Get an existing Firewall by its identifier.
func (fw *FirewallsServiceOp) Get(ctx context.Context, fID string) (*Firewall, *Response, error) {
	path := path.Join(firewallsBasePath, fID)

	req, err := fw.client.NewRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(firewallRoot)
	resp, err := fw.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}

	return root.Firewall, resp, err
}

// Create a new Firewall with a given configuration.
func (fw *FirewallsServiceOp) Create(ctx context.Context, fr *FirewallRequest) (*Firewall, *Response, error) {
	req, err := fw.client.NewRequest(ctx, "POST", firewallsBasePath, fr)
	if err != nil {
		return nil, nil, err
	}

	root := new(firewallRoot)
	resp, err := fw.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}

	return root.Firewall, resp, err
}

// Update an existing Firewall with new configuration.
func (fw *FirewallsServiceOp) Update(ctx context.Context, fID string, fr *FirewallRequest) (*Firewall, *Response, error) {
	path := path.Join(firewallsBasePath, fID)

	req, err := fw.client.NewRequest(ctx, "PUT", path, fr)
	if err != nil {
		return nil, nil, err
	}

	root := new(firewallRoot)
	resp, err := fw.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}

	return root.Firewall, resp, err
}

// Delete a Firewall by its identifier.
func (fw *FirewallsServiceOp) Delete(ctx context.Context, fID string) (*Response, error) {
	path := path.Join(firewallsBasePath, fID)
	return fw.createAndDoReq(ctx, "DELETE", path, nil)
}

// List Firewalls.
func (fw *FirewallsServiceOp) List(ctx context.Context, opt *ListOptions) ([]Firewall, *Response, error) {
	path, err := addOptions(firewallsBasePath, opt)
	if err != nil {
		return nil, nil, err
	}

	return fw.listHelper(ctx, path)
}

// ListByDroplet Firewalls.
func (fw *FirewallsServiceOp) ListByDroplet(ctx context.Context, dID int, opt *ListOptions) ([]Firewall, *Response, error) {
	basePath := path.Join(dropletBasePath, strconv.Itoa(dID), "firewalls")
	path, err := addOptions(basePath, opt)
	if err != nil {
		return nil, nil, err
	}

	return fw.listHelper(ctx, path)
}

// AddDroplets to a Firewall.
func (fw *FirewallsServiceOp) AddDroplets(ctx context.Context, fID string, dropletIDs ...int) (*Response, error) {
	path := path.Join(firewallsBasePath, fID, "droplets")
	return fw.createAndDoReq(ctx, "POST", path, &dropletsRequest{IDs: dropletIDs})
}

// RemoveDroplets from a Firewall.
func (fw *FirewallsServiceOp) RemoveDroplets(ctx context.Context, fID string, dropletIDs ...int) (*Response, error) {
	path := path.Join(firewallsBasePath, fID, "droplets")
	return fw.createAndDoReq(ctx, "DELETE", path, &dropletsRequest{IDs: dropletIDs})
}

// AddTags to a Firewall.
func (fw *FirewallsServiceOp) AddTags(ctx context.Context, fID string, tags ...string) (*Response, error) {
	path := path.Join(firewallsBasePath, fID, "tags")
	return fw.createAndDoReq(ctx, "POST", path, &tagsRequest{Tags: tags})
}

// RemoveTags from a Firewall.
func (fw *FirewallsServiceOp) RemoveTags(ctx context.Context, fID string, tags ...string) (*Response, error) {
	path := path.Join(firewallsBasePath, fID, "tags")
	return fw.createAndDoReq(ctx, "DELETE", path, &tagsRequest{Tags: tags})
}

// AddRules to a Firewall.
func (fw *FirewallsServiceOp) AddRules(ctx context.Context, fID string, rr *FirewallRulesRequest) (*Response, error) {
	path := path.Join(firewallsBasePath, fID, "rules")
	return fw.createAndDoReq(ctx, "POST", path, rr)
}

// RemoveRules from a Firewall.
func (fw *FirewallsServiceOp) RemoveRules(ctx context.Context, fID string, rr *FirewallRulesRequest) (*Response, error) {
	path := path.Join(firewallsBasePath, fID, "rules")
	return fw.createAndDoReq(ctx, "DELETE", path, rr)
}

type dropletsRequest struct {
	IDs []int `json:"droplet_ids"`
}

type tagsRequest struct {
	Tags []string `json:"tags"`
}

type firewallRoot struct {
	Firewall *Firewall `json:"firewall"`
}

type firewallsRoot struct {
	Firewalls []Firewall `json:"firewalls"`
	Links     *Links     `json:"links"`
}

func (fw *FirewallsServiceOp) createAndDoReq(ctx context.Context, method, path string, v interface{}) (*Response, error) {
	req, err := fw.client.NewRequest(ctx, method, path, v)
	if err != nil {
		return nil, err
	}

	return fw.client.Do(ctx, req, nil)
}

func (fw *FirewallsServiceOp) listHelper(ctx context.Context, path string) ([]Firewall, *Response, error) {
	req, err := fw.client.NewRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(firewallsRoot)
	resp, err := fw.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}
	if l := root.Links; l != nil {
		resp.Links = l
	}

	return root.Firewalls, resp, err
}
