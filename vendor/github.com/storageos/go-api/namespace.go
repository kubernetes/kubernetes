package storageos

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/url"

	"github.com/storageos/go-api/types"
)

var (

	// NamespaceAPIPrefix is a partial path to the HTTP endpoint.
	NamespaceAPIPrefix = "namespaces"

	// ErrNoSuchNamespace is the error returned when the namespace does not exist.
	ErrNoSuchNamespace = errors.New("no such namespace")

	// ErrNamespaceInUse is the error returned when the namespace requested to be removed is still in use.
	ErrNamespaceInUse = errors.New("namespace in use and cannot be removed")
)

// NamespaceList returns the list of available namespaces.
func (c *Client) NamespaceList(opts types.ListOptions) ([]*types.Namespace, error) {
	listOpts := doOptions{
		fieldSelector: opts.FieldSelector,
		labelSelector: opts.LabelSelector,
		namespace:     opts.Namespace,
		context:       opts.Context,
	}

	if opts.LabelSelector != "" {
		query := url.Values{}
		query.Add("labelSelector", opts.LabelSelector)
		listOpts.values = query
	}

	resp, err := c.do("GET", NamespaceAPIPrefix, listOpts)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var namespaces []*types.Namespace
	if err := json.NewDecoder(resp.Body).Decode(&namespaces); err != nil {
		return nil, err
	}
	return namespaces, nil
}

// Namespace returns a namespace by its reference.
func (c *Client) Namespace(ref string) (*types.Namespace, error) {
	resp, err := c.do("GET", NamespaceAPIPrefix+"/"+ref, doOptions{})
	if err != nil {
		if e, ok := err.(*Error); ok && e.Status == http.StatusNotFound {
			return nil, ErrNoSuchNamespace
		}
		return nil, err
	}
	defer resp.Body.Close()
	var namespace types.Namespace
	if err := json.NewDecoder(resp.Body).Decode(&namespace); err != nil {
		return nil, err
	}
	return &namespace, nil
}

// NamespaceCreate creates a namespace on the server and returns the new object.
func (c *Client) NamespaceCreate(opts types.NamespaceCreateOptions) (*types.Namespace, error) {
	resp, err := c.do("POST", NamespaceAPIPrefix, doOptions{
		data:    opts,
		context: opts.Context,
	})
	if err != nil {
		return nil, err
	}
	var namespace types.Namespace
	if err := json.NewDecoder(resp.Body).Decode(&namespace); err != nil {
		return nil, err
	}
	return &namespace, nil
}

// NamespaceUpdate updates a namespace on the server and returns the updated object.
func (c *Client) NamespaceUpdate(opts types.NamespaceCreateOptions) (*types.Namespace, error) {
	resp, err := c.do("PUT", NamespaceAPIPrefix+"/"+opts.Name, doOptions{
		data:    opts,
		context: opts.Context,
	})
	if err != nil {
		return nil, err
	}
	var namespace types.Namespace
	if err := json.NewDecoder(resp.Body).Decode(&namespace); err != nil {
		return nil, err
	}
	return &namespace, nil
}

// NamespaceDelete removes a namespace by its reference.
func (c *Client) NamespaceDelete(opts types.DeleteOptions) error {
	deleteOpts := doOptions{
		force:   opts.Force,
		context: opts.Context,
	}
	resp, err := c.do("DELETE", NamespaceAPIPrefix+"/"+opts.Name, deleteOpts)
	if err != nil {
		if e, ok := err.(*Error); ok {
			if e.Status == http.StatusNotFound {
				return ErrNoSuchNamespace
			}
			if e.Status == http.StatusConflict {
				return ErrNamespaceInUse
			}

			// namespace can't be deleted yet, unless force is supplied
			if e.Status == http.StatusPreconditionFailed {
				return err
			}
		}
		return err
	}
	defer resp.Body.Close()
	return nil
}
