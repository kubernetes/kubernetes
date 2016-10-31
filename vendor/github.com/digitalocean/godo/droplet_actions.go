package godo

import (
	"fmt"
	"net/url"
)

// ActionRequest reprents DigitalOcean Action Request
type ActionRequest map[string]interface{}

// DropletActionsService is an interface for interfacing with the droplet actions
// endpoints of the DigitalOcean API
// See: https://developers.digitalocean.com/documentation/v2#droplet-actions
type DropletActionsService interface {
	Shutdown(int) (*Action, *Response, error)
	ShutdownByTag(string) (*Action, *Response, error)
	PowerOff(int) (*Action, *Response, error)
	PowerOffByTag(string) (*Action, *Response, error)
	PowerOn(int) (*Action, *Response, error)
	PowerOnByTag(string) (*Action, *Response, error)
	PowerCycle(int) (*Action, *Response, error)
	PowerCycleByTag(string) (*Action, *Response, error)
	Reboot(int) (*Action, *Response, error)
	Restore(int, int) (*Action, *Response, error)
	Resize(int, string, bool) (*Action, *Response, error)
	Rename(int, string) (*Action, *Response, error)
	Snapshot(int, string) (*Action, *Response, error)
	SnapshotByTag(string, string) (*Action, *Response, error)
	EnableBackups(int) (*Action, *Response, error)
	EnableBackupsByTag(string) (*Action, *Response, error)
	DisableBackups(int) (*Action, *Response, error)
	DisableBackupsByTag(string) (*Action, *Response, error)
	PasswordReset(int) (*Action, *Response, error)
	RebuildByImageID(int, int) (*Action, *Response, error)
	RebuildByImageSlug(int, string) (*Action, *Response, error)
	ChangeKernel(int, int) (*Action, *Response, error)
	EnableIPv6(int) (*Action, *Response, error)
	EnableIPv6ByTag(string) (*Action, *Response, error)
	EnablePrivateNetworking(int) (*Action, *Response, error)
	EnablePrivateNetworkingByTag(string) (*Action, *Response, error)
	Upgrade(int) (*Action, *Response, error)
	Get(int, int) (*Action, *Response, error)
	GetByURI(string) (*Action, *Response, error)
}

// DropletActionsServiceOp handles communication with the droplet action related
// methods of the DigitalOcean API.
type DropletActionsServiceOp struct {
	client *Client
}

var _ DropletActionsService = &DropletActionsServiceOp{}

// Shutdown a Droplet
func (s *DropletActionsServiceOp) Shutdown(id int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "shutdown"}
	return s.doAction(id, request)
}

// Shutdown Droplets by Tag
func (s *DropletActionsServiceOp) ShutdownByTag(tag string) (*Action, *Response, error) {
	request := &ActionRequest{"type": "shutdown"}
	return s.doActionByTag(tag, request)
}

// PowerOff a Droplet
func (s *DropletActionsServiceOp) PowerOff(id int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "power_off"}
	return s.doAction(id, request)
}

// PowerOff a Droplet by Tag
func (s *DropletActionsServiceOp) PowerOffByTag(tag string) (*Action, *Response, error) {
	request := &ActionRequest{"type": "power_off"}
	return s.doActionByTag(tag, request)
}

// PowerOn a Droplet
func (s *DropletActionsServiceOp) PowerOn(id int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "power_on"}
	return s.doAction(id, request)
}

// PowerOn a Droplet by Tag
func (s *DropletActionsServiceOp) PowerOnByTag(tag string) (*Action, *Response, error) {
	request := &ActionRequest{"type": "power_on"}
	return s.doActionByTag(tag, request)
}

// PowerCycle a Droplet
func (s *DropletActionsServiceOp) PowerCycle(id int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "power_cycle"}
	return s.doAction(id, request)
}

// PowerCycle a Droplet by Tag
func (s *DropletActionsServiceOp) PowerCycleByTag(tag string) (*Action, *Response, error) {
	request := &ActionRequest{"type": "power_cycle"}
	return s.doActionByTag(tag, request)
}

// Reboot a Droplet
func (s *DropletActionsServiceOp) Reboot(id int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "reboot"}
	return s.doAction(id, request)
}

// Restore an image to a Droplet
func (s *DropletActionsServiceOp) Restore(id, imageID int) (*Action, *Response, error) {
	requestType := "restore"
	request := &ActionRequest{
		"type":  requestType,
		"image": float64(imageID),
	}
	return s.doAction(id, request)
}

// Resize a Droplet
func (s *DropletActionsServiceOp) Resize(id int, sizeSlug string, resizeDisk bool) (*Action, *Response, error) {
	requestType := "resize"
	request := &ActionRequest{
		"type": requestType,
		"size": sizeSlug,
		"disk": resizeDisk,
	}
	return s.doAction(id, request)
}

// Rename a Droplet
func (s *DropletActionsServiceOp) Rename(id int, name string) (*Action, *Response, error) {
	requestType := "rename"
	request := &ActionRequest{
		"type": requestType,
		"name": name,
	}
	return s.doAction(id, request)
}

// Snapshot a Droplet.
func (s *DropletActionsServiceOp) Snapshot(id int, name string) (*Action, *Response, error) {
	requestType := "snapshot"
	request := &ActionRequest{
		"type": requestType,
		"name": name,
	}
	return s.doAction(id, request)
}

// Snapshot a Droplet by Tag
func (s *DropletActionsServiceOp) SnapshotByTag(tag string, name string) (*Action, *Response, error) {
	requestType := "snapshot"
	request := &ActionRequest{
		"type": requestType,
		"name": name,
	}
	return s.doActionByTag(tag, request)
}

// EnableBackups enables backups for a droplet.
func (s *DropletActionsServiceOp) EnableBackups(id int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "enable_backups"}
	return s.doAction(id, request)
}

// EnableBackups enables backups for a droplet by Tag
func (s *DropletActionsServiceOp) EnableBackupsByTag(tag string) (*Action, *Response, error) {
	request := &ActionRequest{"type": "enable_backups"}
	return s.doActionByTag(tag, request)
}

// DisableBackups disables backups for a droplet.
func (s *DropletActionsServiceOp) DisableBackups(id int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "disable_backups"}
	return s.doAction(id, request)
}

// DisableBackups disables backups for a droplet by tag
func (s *DropletActionsServiceOp) DisableBackupsByTag(tag string) (*Action, *Response, error) {
	request := &ActionRequest{"type": "disable_backups"}
	return s.doActionByTag(tag, request)
}

// PasswordReset resets the password for a droplet.
func (s *DropletActionsServiceOp) PasswordReset(id int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "password_reset"}
	return s.doAction(id, request)
}

// RebuildByImageID rebuilds a droplet droplet from an image with a given id.
func (s *DropletActionsServiceOp) RebuildByImageID(id, imageID int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "rebuild", "image": imageID}
	return s.doAction(id, request)
}

// RebuildByImageSlug rebuilds a droplet from an image with a given slug.
func (s *DropletActionsServiceOp) RebuildByImageSlug(id int, slug string) (*Action, *Response, error) {
	request := &ActionRequest{"type": "rebuild", "image": slug}
	return s.doAction(id, request)
}

// ChangeKernel changes the kernel for a droplet.
func (s *DropletActionsServiceOp) ChangeKernel(id, kernelID int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "change_kernel", "kernel": kernelID}
	return s.doAction(id, request)
}

// EnableIPv6 enables IPv6 for a droplet.
func (s *DropletActionsServiceOp) EnableIPv6(id int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "enable_ipv6"}
	return s.doAction(id, request)
}

// EnableIPv6 enables IPv6 for a droplet by Tag
func (s *DropletActionsServiceOp) EnableIPv6ByTag(tag string) (*Action, *Response, error) {
	request := &ActionRequest{"type": "enable_ipv6"}
	return s.doActionByTag(tag, request)
}

// EnablePrivateNetworking enables private networking for a droplet.
func (s *DropletActionsServiceOp) EnablePrivateNetworking(id int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "enable_private_networking"}
	return s.doAction(id, request)
}

// EnablePrivateNetworking enables private networking for a droplet by Tag
func (s *DropletActionsServiceOp) EnablePrivateNetworkingByTag(tag string) (*Action, *Response, error) {
	request := &ActionRequest{"type": "enable_private_networking"}
	return s.doActionByTag(tag, request)
}

// Upgrade a droplet.
func (s *DropletActionsServiceOp) Upgrade(id int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "upgrade"}
	return s.doAction(id, request)
}

func (s *DropletActionsServiceOp) doAction(id int, request *ActionRequest) (*Action, *Response, error) {
	if id < 1 {
		return nil, nil, NewArgError("id", "cannot be less than 1")
	}

	if request == nil {
		return nil, nil, NewArgError("request", "request can't be nil")
	}

	path := dropletActionPath(id)

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

func (s *DropletActionsServiceOp) doActionByTag(tag string, request *ActionRequest) (*Action, *Response, error) {
	if tag == "" {
		return nil, nil, NewArgError("tag", "cannot be empty")
	}

	if request == nil {
		return nil, nil, NewArgError("request", "request can't be nil")
	}

	path := dropletActionPathByTag(tag)

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

// Get an action for a particular droplet by id.
func (s *DropletActionsServiceOp) Get(dropletID, actionID int) (*Action, *Response, error) {
	if dropletID < 1 {
		return nil, nil, NewArgError("dropletID", "cannot be less than 1")
	}

	if actionID < 1 {
		return nil, nil, NewArgError("actionID", "cannot be less than 1")
	}

	path := fmt.Sprintf("%s/%d", dropletActionPath(dropletID), actionID)
	return s.get(path)
}

// GetByURI gets an action for a particular droplet by id.
func (s *DropletActionsServiceOp) GetByURI(rawurl string) (*Action, *Response, error) {
	u, err := url.Parse(rawurl)
	if err != nil {
		return nil, nil, err
	}

	return s.get(u.Path)

}

func (s *DropletActionsServiceOp) get(path string) (*Action, *Response, error) {
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

func dropletActionPath(dropletID int) string {
	return fmt.Sprintf("v2/droplets/%d/actions", dropletID)
}

func dropletActionPathByTag(tag string) string {
	return fmt.Sprintf("v2/droplets/actions?tag_name=%s", tag)
}
