package godo

import "fmt"

// FloatingIPActionsService is an interface for interfacing with the
// floating IPs actions endpoints of the Digital Ocean API.
// See: https://developers.digitalocean.com/documentation/v2#floating-ips-action
type FloatingIPActionsService interface {
	Assign(ip string, dropletID int) (*Action, *Response, error)
	Unassign(ip string) (*Action, *Response, error)
	Get(ip string, actionID int) (*Action, *Response, error)
	List(ip string, opt *ListOptions) ([]Action, *Response, error)
}

// FloatingIPActionsServiceOp handles communication with the floating IPs
// action related methods of the DigitalOcean API.
type FloatingIPActionsServiceOp struct {
	client *Client
}

// Assign a floating IP to a droplet.
func (s *FloatingIPActionsServiceOp) Assign(ip string, dropletID int) (*Action, *Response, error) {
	request := &ActionRequest{
		"type":       "assign",
		"droplet_id": dropletID,
	}
	return s.doAction(ip, request)
}

// Unassign a floating IP from the droplet it is currently assigned to.
func (s *FloatingIPActionsServiceOp) Unassign(ip string) (*Action, *Response, error) {
	request := &ActionRequest{"type": "unassign"}
	return s.doAction(ip, request)
}

// Get an action for a particular floating IP by id.
func (s *FloatingIPActionsServiceOp) Get(ip string, actionID int) (*Action, *Response, error) {
	path := fmt.Sprintf("%s/%d", floatingIPActionPath(ip), actionID)
	return s.get(path)
}

// List the actions for a particular floating IP.
func (s *FloatingIPActionsServiceOp) List(ip string, opt *ListOptions) ([]Action, *Response, error) {
	path := floatingIPActionPath(ip)
	path, err := addOptions(path, opt)
	if err != nil {
		return nil, nil, err
	}

	return s.list(path)
}

func (s *FloatingIPActionsServiceOp) doAction(ip string, request *ActionRequest) (*Action, *Response, error) {
	path := floatingIPActionPath(ip)

	req, err := s.client.NewRequest("POST", path, request)
	if err != nil {
		return nil, nil, err
	}

	root := new(actionRoot)
	resp, err := s.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}

	return &root.Event, resp, err
}

func (s *FloatingIPActionsServiceOp) get(path string) (*Action, *Response, error) {
	req, err := s.client.NewRequest("GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(actionRoot)
	resp, err := s.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}

	return &root.Event, resp, err
}

func (s *FloatingIPActionsServiceOp) list(path string) ([]Action, *Response, error) {
	req, err := s.client.NewRequest("GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(actionsRoot)
	resp, err := s.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}
	if l := root.Links; l != nil {
		resp.Links = l
	}

	return root.Actions, resp, err
}

func floatingIPActionPath(ip string) string {
	return fmt.Sprintf("%s/%s/actions", floatingBasePath, ip)
}
