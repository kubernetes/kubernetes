package godo

import "fmt"

// ImageActionsService is an interface for interfacing with the image actions
// endpoints of the DigitalOcean API
// See: https://developers.digitalocean.com/documentation/v2#image-actions
type ImageActionsService interface {
	Get(int, int) (*Action, *Response, error)
	Transfer(int, *ActionRequest) (*Action, *Response, error)
}

// ImageActionsServiceOp handles communition with the image action related methods of the
// DigitalOcean API.
type ImageActionsServiceOp struct {
	client *Client
}

var _ ImageActionsService = &ImageActionsServiceOp{}

// Transfer an image
func (i *ImageActionsServiceOp) Transfer(imageID int, transferRequest *ActionRequest) (*Action, *Response, error) {
	if imageID < 1 {
		return nil, nil, NewArgError("imageID", "cannot be less than 1")
	}

	if transferRequest == nil {
		return nil, nil, NewArgError("transferRequest", "cannot be nil")
	}

	path := fmt.Sprintf("v2/images/%d/actions", imageID)

	req, err := i.client.NewRequest("POST", path, transferRequest)
	if err != nil {
		return nil, nil, err
	}

	root := new(actionRoot)
	resp, err := i.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}

	return &root.Event, resp, err
}

// Get an action for a particular image by id.
func (i *ImageActionsServiceOp) Get(imageID, actionID int) (*Action, *Response, error) {
	if imageID < 1 {
		return nil, nil, NewArgError("imageID", "cannot be less than 1")
	}

	if actionID < 1 {
		return nil, nil, NewArgError("actionID", "cannot be less than 1")
	}

	path := fmt.Sprintf("v2/images/%d/actions/%d", imageID, actionID)

	req, err := i.client.NewRequest("GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(actionRoot)
	resp, err := i.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}

	return &root.Event, resp, err
}
