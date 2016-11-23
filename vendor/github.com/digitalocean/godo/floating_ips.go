package godo

import "fmt"

const floatingBasePath = "v2/floating_ips"

// FloatingIPsService is an interface for interfacing with the floating IPs
// endpoints of the Digital Ocean API.
// See: https://developers.digitalocean.com/documentation/v2#floating-ips
type FloatingIPsService interface {
	List(*ListOptions) ([]FloatingIP, *Response, error)
	Get(string) (*FloatingIP, *Response, error)
	Create(*FloatingIPCreateRequest) (*FloatingIP, *Response, error)
	Delete(string) (*Response, error)
}

// FloatingIPsServiceOp handles communication with the floating IPs related methods of the
// DigitalOcean API.
type FloatingIPsServiceOp struct {
	client *Client
}

var _ FloatingIPsService = &FloatingIPsServiceOp{}

// FloatingIP represents a Digital Ocean floating IP.
type FloatingIP struct {
	Region  *Region  `json:"region"`
	Droplet *Droplet `json:"droplet"`
	IP      string   `json:"ip"`
}

func (f FloatingIP) String() string {
	return Stringify(f)
}

type floatingIPsRoot struct {
	FloatingIPs []FloatingIP `json:"floating_ips"`
	Links       *Links       `json:"links"`
}

type floatingIPRoot struct {
	FloatingIP *FloatingIP `json:"floating_ip"`
	Links      *Links      `json:"links,omitempty"`
}

// FloatingIPCreateRequest represents a request to create a floating IP.
// If DropletID is not empty, the floating IP will be assigned to the
// droplet.
type FloatingIPCreateRequest struct {
	Region    string `json:"region"`
	DropletID int    `json:"droplet_id,omitempty"`
}

// List all floating IPs.
func (f *FloatingIPsServiceOp) List(opt *ListOptions) ([]FloatingIP, *Response, error) {
	path := floatingBasePath
	path, err := addOptions(path, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := f.client.NewRequest("GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(floatingIPsRoot)
	resp, err := f.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}
	if l := root.Links; l != nil {
		resp.Links = l
	}

	return root.FloatingIPs, resp, err
}

// Get an individual floating IP.
func (f *FloatingIPsServiceOp) Get(ip string) (*FloatingIP, *Response, error) {
	path := fmt.Sprintf("%s/%s", floatingBasePath, ip)

	req, err := f.client.NewRequest("GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(floatingIPRoot)
	resp, err := f.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}

	return root.FloatingIP, resp, err
}

// Create a floating IP. If the DropletID field of the request is not empty,
// the floating IP will also be assigned to the droplet.
func (f *FloatingIPsServiceOp) Create(createRequest *FloatingIPCreateRequest) (*FloatingIP, *Response, error) {
	path := floatingBasePath

	req, err := f.client.NewRequest("POST", path, createRequest)
	if err != nil {
		return nil, nil, err
	}

	root := new(floatingIPRoot)
	resp, err := f.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}
	if l := root.Links; l != nil {
		resp.Links = l
	}

	return root.FloatingIP, resp, err
}

// Delete a floating IP.
func (f *FloatingIPsServiceOp) Delete(ip string) (*Response, error) {
	path := fmt.Sprintf("%s/%s", floatingBasePath, ip)

	req, err := f.client.NewRequest("DELETE", path, nil)
	if err != nil {
		return nil, err
	}

	resp, err := f.client.Do(req, nil)

	return resp, err
}
