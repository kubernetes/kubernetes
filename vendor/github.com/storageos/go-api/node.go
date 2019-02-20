package storageos

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/url"

	"github.com/storageos/go-api/types"
)

var (

	// NodeAPIPrefix is a partial path to the HTTP endpoint.
	NodeAPIPrefix = "nodes"

	// ErrNoSuchNode is the error returned when the node does not exist.
	ErrNoSuchNode = errors.New("no such node")

	// ErrNodeInUse is the error returned when the node requested to be removed is still in use.
	ErrNodeInUse = errors.New("node in use and cannot be removed")
)

// NodeList returns the list of available nodes.
func (c *Client) NodeList(opts types.ListOptions) ([]*types.Node, error) {
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

	resp, err := c.do("GET", NodeAPIPrefix, listOpts)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var nodes []*types.Node
	if err := json.NewDecoder(resp.Body).Decode(&nodes); err != nil {
		return nil, err
	}
	return nodes, nil
}

// Node returns a node by its reference.
func (c *Client) Node(ref string) (*types.Node, error) {

	resp, err := c.do("GET", NodeAPIPrefix+"/"+ref, doOptions{})
	if err != nil {
		if e, ok := err.(*Error); ok && e.Status == http.StatusNotFound {
			return nil, ErrNoSuchNode
		}
		return nil, err
	}
	defer resp.Body.Close()
	var node types.Node
	if err := json.NewDecoder(resp.Body).Decode(&node); err != nil {
		return nil, err
	}
	return &node, nil
}

// NodeUpdate updates a node on the server.
func (c *Client) NodeUpdate(opts types.NodeUpdateOptions) (*types.Node, error) {
	ref := opts.Name
	if IsUUID(opts.ID) {
		ref = opts.ID
	}
	resp, err := c.do("PUT", NodeAPIPrefix+"/"+ref, doOptions{
		data:    opts,
		context: opts.Context,
	})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var node types.Node
	if err := json.NewDecoder(resp.Body).Decode(&node); err != nil {
		return nil, err
	}
	return &node, nil
}

// NodeDelete removes a node by its reference.
func (c *Client) NodeDelete(opts types.DeleteOptions) error {
	deleteOpts := doOptions{
		namespace: opts.Namespace,
		force:     opts.Force,
		context:   opts.Context,
	}
	resp, err := c.do("DELETE", NodeAPIPrefix+"/"+opts.Name, deleteOpts)
	if err != nil {
		if e, ok := err.(*Error); ok {
			if e.Status == http.StatusNotFound {
				return ErrNoSuchNode
			}
			if e.Status == http.StatusConflict {
				return ErrNodeInUse
			}
		}
		return err
	}
	defer resp.Body.Close()
	return nil
}
