package godo

import "fmt"

// StorageActionsService is an interface for interfacing with the
// storage actions endpoints of the Digital Ocean API.
// See: https://developers.digitalocean.com/documentation/v2#storage-actions
type StorageActionsService interface {
	Attach(volumeID string, dropletID int) (*Action, *Response, error)
	Detach(volumeID string) (*Action, *Response, error)
	DetachByDropletID(volumeID string, dropletID int) (*Action, *Response, error)
	Get(volumeID string, actionID int) (*Action, *Response, error)
	List(volumeID string, opt *ListOptions) ([]Action, *Response, error)
	Resize(volumeID string, sizeGigabytes int, regionSlug string) (*Action, *Response, error)
}

// StorageActionsServiceOp handles communication with the storage volumes
// action related methods of the DigitalOcean API.
type StorageActionsServiceOp struct {
	client *Client
}

// StorageAttachment represents the attachement of a block storage
// volume to a specific droplet under the device name.
type StorageAttachment struct {
	DropletID int `json:"droplet_id"`
}

// Attach a storage volume to a droplet.
func (s *StorageActionsServiceOp) Attach(volumeID string, dropletID int) (*Action, *Response, error) {
	request := &ActionRequest{
		"type":       "attach",
		"droplet_id": dropletID,
	}
	return s.doAction(volumeID, request)
}

// Detach a storage volume from a droplet.
func (s *StorageActionsServiceOp) Detach(volumeID string) (*Action, *Response, error) {
	request := &ActionRequest{
		"type": "detach",
	}
	return s.doAction(volumeID, request)
}

// Detach a storage volume from a droplet by droplet ID.
func (s *StorageActionsServiceOp) DetachByDropletID(volumeID string, dropletID int) (*Action, *Response, error) {
	request := &ActionRequest{
		"type":       "detach",
		"droplet_id": dropletID,
	}
	return s.doAction(volumeID, request)
}

// Get an action for a particular storage volume by id.
func (s *StorageActionsServiceOp) Get(volumeID string, actionID int) (*Action, *Response, error) {
	path := fmt.Sprintf("%s/%d", storageAllocationActionPath(volumeID), actionID)
	return s.get(path)
}

// List the actions for a particular storage volume.
func (s *StorageActionsServiceOp) List(volumeID string, opt *ListOptions) ([]Action, *Response, error) {
	path := storageAllocationActionPath(volumeID)
	path, err := addOptions(path, opt)
	if err != nil {
		return nil, nil, err
	}

	return s.list(path)
}

// Resize a storage volume.
func (s *StorageActionsServiceOp) Resize(volumeID string, sizeGigabytes int, regionSlug string) (*Action, *Response, error) {
	request := &ActionRequest{
		"type":           "resize",
		"size_gigabytes": sizeGigabytes,
		"region":         regionSlug,
	}
	return s.doAction(volumeID, request)
}

func (s *StorageActionsServiceOp) doAction(volumeID string, request *ActionRequest) (*Action, *Response, error) {
	path := storageAllocationActionPath(volumeID)

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

func (s *StorageActionsServiceOp) get(path string) (*Action, *Response, error) {
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

func (s *StorageActionsServiceOp) list(path string) ([]Action, *Response, error) {
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

func storageAllocationActionPath(volumeID string) string {
	return fmt.Sprintf("%s/%s/actions", storageAllocPath, volumeID)
}
