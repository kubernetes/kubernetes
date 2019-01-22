package storageos

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/url"

	"github.com/storageos/go-api/types"
)

var (

	// ControllerAPIPrefix is a partial path to the HTTP endpoint.
	ControllerAPIPrefix = "controllers"

	// ErrNoSuchController is the error returned when the controller does not exist.
	ErrNoSuchController = errors.New("no such controller")

	// ErrControllerInUse is the error returned when the controller requested to be removed is still in use.
	ErrControllerInUse = errors.New("controller in use and cannot be removed")
)

// ControllerList returns the list of available controllers.
func (c *Client) ControllerList(opts types.ListOptions) ([]*types.Controller, error) {
	listOpts := doOptions{
		fieldSelector: opts.FieldSelector,
		labelSelector: opts.LabelSelector,
		context:       opts.Context,
	}

	if opts.LabelSelector != "" {
		query := url.Values{}
		query.Add("labelSelector", opts.LabelSelector)
		listOpts.values = query
	}

	resp, err := c.do("GET", ControllerAPIPrefix, listOpts)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var controllers []*types.Controller
	if err := json.NewDecoder(resp.Body).Decode(&controllers); err != nil {
		return nil, err
	}
	return controllers, nil
}

// Controller returns a controller by its reference.
func (c *Client) Controller(ref string) (*types.Controller, error) {

	resp, err := c.do("GET", ControllerAPIPrefix+"/"+ref, doOptions{})
	if err != nil {
		if e, ok := err.(*Error); ok && e.Status == http.StatusNotFound {
			return nil, ErrNoSuchController
		}
		return nil, err
	}
	defer resp.Body.Close()
	var controller types.Controller
	if err := json.NewDecoder(resp.Body).Decode(&controller); err != nil {
		return nil, err
	}
	return &controller, nil
}

// ControllerUpdate updates a controller on the server.
func (c *Client) ControllerUpdate(opts types.ControllerUpdateOptions) (*types.Controller, error) {
	ref := opts.Name
	if IsUUID(opts.ID) {
		ref = opts.ID
	}
	resp, err := c.do("PUT", ControllerAPIPrefix+"/"+ref, doOptions{
		data:    opts,
		context: opts.Context,
	})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var controller types.Controller
	if err := json.NewDecoder(resp.Body).Decode(&controller); err != nil {
		return nil, err
	}
	return &controller, nil
}

// ControllerDelete removes a controller by its reference.
func (c *Client) ControllerDelete(opts types.DeleteOptions) error {
	deleteOpts := doOptions{
		namespace: opts.Namespace,
		force:     opts.Force,
		context:   opts.Context,
	}
	resp, err := c.do("DELETE", ControllerAPIPrefix+"/"+opts.Name, deleteOpts)
	if err != nil {
		if e, ok := err.(*Error); ok {
			if e.Status == http.StatusNotFound {
				return ErrNoSuchController
			}
			if e.Status == http.StatusConflict {
				return ErrControllerInUse
			}
		}
		return err
	}
	defer resp.Body.Close()
	return nil
}
