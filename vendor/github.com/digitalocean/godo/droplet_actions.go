package godo

import (
	"fmt"
	"net/url"

	"github.com/digitalocean/godo/context"
)

// ActionRequest reprents DigitalOcean Action Request
type ActionRequest map[string]interface{}

// DropletActionsService is an interface for interfacing with the Droplet actions
// endpoints of the DigitalOcean API
// See: https://developers.digitalocean.com/documentation/v2#droplet-actions
type DropletActionsService interface {
	Shutdown(context.Context, int) (*Action, *Response, error)
	ShutdownByTag(context.Context, string) ([]Action, *Response, error)
	PowerOff(context.Context, int) (*Action, *Response, error)
	PowerOffByTag(context.Context, string) ([]Action, *Response, error)
	PowerOn(context.Context, int) (*Action, *Response, error)
	PowerOnByTag(context.Context, string) ([]Action, *Response, error)
	PowerCycle(context.Context, int) (*Action, *Response, error)
	PowerCycleByTag(context.Context, string) ([]Action, *Response, error)
	Reboot(context.Context, int) (*Action, *Response, error)
	Restore(context.Context, int, int) (*Action, *Response, error)
	Resize(context.Context, int, string, bool) (*Action, *Response, error)
	Rename(context.Context, int, string) (*Action, *Response, error)
	Snapshot(context.Context, int, string) (*Action, *Response, error)
	SnapshotByTag(context.Context, string, string) ([]Action, *Response, error)
	EnableBackups(context.Context, int) (*Action, *Response, error)
	EnableBackupsByTag(context.Context, string) ([]Action, *Response, error)
	DisableBackups(context.Context, int) (*Action, *Response, error)
	DisableBackupsByTag(context.Context, string) ([]Action, *Response, error)
	PasswordReset(context.Context, int) (*Action, *Response, error)
	RebuildByImageID(context.Context, int, int) (*Action, *Response, error)
	RebuildByImageSlug(context.Context, int, string) (*Action, *Response, error)
	ChangeKernel(context.Context, int, int) (*Action, *Response, error)
	EnableIPv6(context.Context, int) (*Action, *Response, error)
	EnableIPv6ByTag(context.Context, string) ([]Action, *Response, error)
	EnablePrivateNetworking(context.Context, int) (*Action, *Response, error)
	EnablePrivateNetworkingByTag(context.Context, string) ([]Action, *Response, error)
	Upgrade(context.Context, int) (*Action, *Response, error)
	Get(context.Context, int, int) (*Action, *Response, error)
	GetByURI(context.Context, string) (*Action, *Response, error)
}

// DropletActionsServiceOp handles communication with the Droplet action related
// methods of the DigitalOcean API.
type DropletActionsServiceOp struct {
	client *Client
}

var _ DropletActionsService = &DropletActionsServiceOp{}

// Shutdown a Droplet
func (s *DropletActionsServiceOp) Shutdown(ctx context.Context, id int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "shutdown"}
	return s.doAction(ctx, id, request)
}

// ShutdownByTag shuts down Droplets matched by a Tag.
func (s *DropletActionsServiceOp) ShutdownByTag(ctx context.Context, tag string) ([]Action, *Response, error) {
	request := &ActionRequest{"type": "shutdown"}
	return s.doActionByTag(ctx, tag, request)
}

// PowerOff a Droplet
func (s *DropletActionsServiceOp) PowerOff(ctx context.Context, id int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "power_off"}
	return s.doAction(ctx, id, request)
}

// PowerOffByTag powers off Droplets matched by a Tag.
func (s *DropletActionsServiceOp) PowerOffByTag(ctx context.Context, tag string) ([]Action, *Response, error) {
	request := &ActionRequest{"type": "power_off"}
	return s.doActionByTag(ctx, tag, request)
}

// PowerOn a Droplet
func (s *DropletActionsServiceOp) PowerOn(ctx context.Context, id int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "power_on"}
	return s.doAction(ctx, id, request)
}

// PowerOnByTag powers on Droplets matched by a Tag.
func (s *DropletActionsServiceOp) PowerOnByTag(ctx context.Context, tag string) ([]Action, *Response, error) {
	request := &ActionRequest{"type": "power_on"}
	return s.doActionByTag(ctx, tag, request)
}

// PowerCycle a Droplet
func (s *DropletActionsServiceOp) PowerCycle(ctx context.Context, id int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "power_cycle"}
	return s.doAction(ctx, id, request)
}

// PowerCycleByTag power cycles Droplets matched by a Tag.
func (s *DropletActionsServiceOp) PowerCycleByTag(ctx context.Context, tag string) ([]Action, *Response, error) {
	request := &ActionRequest{"type": "power_cycle"}
	return s.doActionByTag(ctx, tag, request)
}

// Reboot a Droplet
func (s *DropletActionsServiceOp) Reboot(ctx context.Context, id int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "reboot"}
	return s.doAction(ctx, id, request)
}

// Restore an image to a Droplet
func (s *DropletActionsServiceOp) Restore(ctx context.Context, id, imageID int) (*Action, *Response, error) {
	requestType := "restore"
	request := &ActionRequest{
		"type":  requestType,
		"image": float64(imageID),
	}
	return s.doAction(ctx, id, request)
}

// Resize a Droplet
func (s *DropletActionsServiceOp) Resize(ctx context.Context, id int, sizeSlug string, resizeDisk bool) (*Action, *Response, error) {
	requestType := "resize"
	request := &ActionRequest{
		"type": requestType,
		"size": sizeSlug,
		"disk": resizeDisk,
	}
	return s.doAction(ctx, id, request)
}

// Rename a Droplet
func (s *DropletActionsServiceOp) Rename(ctx context.Context, id int, name string) (*Action, *Response, error) {
	requestType := "rename"
	request := &ActionRequest{
		"type": requestType,
		"name": name,
	}
	return s.doAction(ctx, id, request)
}

// Snapshot a Droplet.
func (s *DropletActionsServiceOp) Snapshot(ctx context.Context, id int, name string) (*Action, *Response, error) {
	requestType := "snapshot"
	request := &ActionRequest{
		"type": requestType,
		"name": name,
	}
	return s.doAction(ctx, id, request)
}

// SnapshotByTag snapshots Droplets matched by a Tag.
func (s *DropletActionsServiceOp) SnapshotByTag(ctx context.Context, tag string, name string) ([]Action, *Response, error) {
	requestType := "snapshot"
	request := &ActionRequest{
		"type": requestType,
		"name": name,
	}
	return s.doActionByTag(ctx, tag, request)
}

// EnableBackups enables backups for a Droplet.
func (s *DropletActionsServiceOp) EnableBackups(ctx context.Context, id int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "enable_backups"}
	return s.doAction(ctx, id, request)
}

// EnableBackupsByTag enables backups for Droplets matched by a Tag.
func (s *DropletActionsServiceOp) EnableBackupsByTag(ctx context.Context, tag string) ([]Action, *Response, error) {
	request := &ActionRequest{"type": "enable_backups"}
	return s.doActionByTag(ctx, tag, request)
}

// DisableBackups disables backups for a Droplet.
func (s *DropletActionsServiceOp) DisableBackups(ctx context.Context, id int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "disable_backups"}
	return s.doAction(ctx, id, request)
}

// DisableBackupsByTag disables backups for Droplet matched by a Tag.
func (s *DropletActionsServiceOp) DisableBackupsByTag(ctx context.Context, tag string) ([]Action, *Response, error) {
	request := &ActionRequest{"type": "disable_backups"}
	return s.doActionByTag(ctx, tag, request)
}

// PasswordReset resets the password for a Droplet.
func (s *DropletActionsServiceOp) PasswordReset(ctx context.Context, id int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "password_reset"}
	return s.doAction(ctx, id, request)
}

// RebuildByImageID rebuilds a Droplet from an image with a given id.
func (s *DropletActionsServiceOp) RebuildByImageID(ctx context.Context, id, imageID int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "rebuild", "image": imageID}
	return s.doAction(ctx, id, request)
}

// RebuildByImageSlug rebuilds a Droplet from an Image matched by a given Slug.
func (s *DropletActionsServiceOp) RebuildByImageSlug(ctx context.Context, id int, slug string) (*Action, *Response, error) {
	request := &ActionRequest{"type": "rebuild", "image": slug}
	return s.doAction(ctx, id, request)
}

// ChangeKernel changes the kernel for a Droplet.
func (s *DropletActionsServiceOp) ChangeKernel(ctx context.Context, id, kernelID int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "change_kernel", "kernel": kernelID}
	return s.doAction(ctx, id, request)
}

// EnableIPv6 enables IPv6 for a Droplet.
func (s *DropletActionsServiceOp) EnableIPv6(ctx context.Context, id int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "enable_ipv6"}
	return s.doAction(ctx, id, request)
}

// EnableIPv6ByTag enables IPv6 for Droplets matched by a Tag.
func (s *DropletActionsServiceOp) EnableIPv6ByTag(ctx context.Context, tag string) ([]Action, *Response, error) {
	request := &ActionRequest{"type": "enable_ipv6"}
	return s.doActionByTag(ctx, tag, request)
}

// EnablePrivateNetworking enables private networking for a Droplet.
func (s *DropletActionsServiceOp) EnablePrivateNetworking(ctx context.Context, id int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "enable_private_networking"}
	return s.doAction(ctx, id, request)
}

// EnablePrivateNetworkingByTag enables private networking for Droplets matched by a Tag.
func (s *DropletActionsServiceOp) EnablePrivateNetworkingByTag(ctx context.Context, tag string) ([]Action, *Response, error) {
	request := &ActionRequest{"type": "enable_private_networking"}
	return s.doActionByTag(ctx, tag, request)
}

// Upgrade a Droplet.
func (s *DropletActionsServiceOp) Upgrade(ctx context.Context, id int) (*Action, *Response, error) {
	request := &ActionRequest{"type": "upgrade"}
	return s.doAction(ctx, id, request)
}

func (s *DropletActionsServiceOp) doAction(ctx context.Context, id int, request *ActionRequest) (*Action, *Response, error) {
	if id < 1 {
		return nil, nil, NewArgError("id", "cannot be less than 1")
	}

	if request == nil {
		return nil, nil, NewArgError("request", "request can't be nil")
	}

	path := dropletActionPath(id)

	req, err := s.client.NewRequest(ctx, "POST", path, request)
	if err != nil {
		return nil, nil, err
	}

	root := new(actionRoot)
	resp, err := s.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}

	return root.Event, resp, err
}

func (s *DropletActionsServiceOp) doActionByTag(ctx context.Context, tag string, request *ActionRequest) ([]Action, *Response, error) {
	if tag == "" {
		return nil, nil, NewArgError("tag", "cannot be empty")
	}

	if request == nil {
		return nil, nil, NewArgError("request", "request can't be nil")
	}

	path := dropletActionPathByTag(tag)

	req, err := s.client.NewRequest(ctx, "POST", path, request)
	if err != nil {
		return nil, nil, err
	}

	root := new(actionsRoot)
	resp, err := s.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}

	return root.Actions, resp, err
}

// Get an action for a particular Droplet by id.
func (s *DropletActionsServiceOp) Get(ctx context.Context, dropletID, actionID int) (*Action, *Response, error) {
	if dropletID < 1 {
		return nil, nil, NewArgError("dropletID", "cannot be less than 1")
	}

	if actionID < 1 {
		return nil, nil, NewArgError("actionID", "cannot be less than 1")
	}

	path := fmt.Sprintf("%s/%d", dropletActionPath(dropletID), actionID)
	return s.get(ctx, path)
}

// GetByURI gets an action for a particular Droplet by id.
func (s *DropletActionsServiceOp) GetByURI(ctx context.Context, rawurl string) (*Action, *Response, error) {
	u, err := url.Parse(rawurl)
	if err != nil {
		return nil, nil, err
	}

	return s.get(ctx, u.Path)

}

func (s *DropletActionsServiceOp) get(ctx context.Context, path string) (*Action, *Response, error) {
	req, err := s.client.NewRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(actionRoot)
	resp, err := s.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}

	return root.Event, resp, err

}

func dropletActionPath(dropletID int) string {
	return fmt.Sprintf("v2/droplets/%d/actions", dropletID)
}

func dropletActionPathByTag(tag string) string {
	return fmt.Sprintf("v2/droplets/actions?tag_name=%s", tag)
}
