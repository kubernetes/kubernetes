package storageos

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/url"

	"github.com/storageos/go-api/types"
)

var (

	// VolumeAPIPrefix is a partial path to the HTTP endpoint.
	VolumeAPIPrefix = "volumes"

	// ErrNoSuchVolume is the error returned when the volume does not exist.
	ErrNoSuchVolume = errors.New("no such volume")

	// ErrVolumeInUse is the error returned when the volume requested to be removed is still in use.
	ErrVolumeInUse = errors.New("volume in use and cannot be removed")
)

// VolumeList returns the list of available volumes.
func (c *Client) VolumeList(opts types.ListOptions) ([]*types.Volume, error) {
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

	resp, err := c.do("GET", VolumeAPIPrefix, listOpts)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var volumes []*types.Volume
	if err := json.NewDecoder(resp.Body).Decode(&volumes); err != nil {
		return nil, err
	}
	return volumes, nil
}

// Volume returns a volume by its reference.
func (c *Client) Volume(namespace string, ref string) (*types.Volume, error) {
	path, err := namespacedRefPath(namespace, VolumeAPIPrefix, ref)
	if err != nil {
		return nil, err
	}
	resp, err := c.do("GET", path, doOptions{})
	if err != nil {
		if e, ok := err.(*Error); ok && e.Status == http.StatusNotFound {
			return nil, ErrNoSuchVolume
		}
		return nil, err
	}
	defer resp.Body.Close()
	var volume types.Volume
	if err := json.NewDecoder(resp.Body).Decode(&volume); err != nil {
		return nil, err
	}
	return &volume, nil
}

// VolumeCreate creates a volume on the server and returns the new object.
func (c *Client) VolumeCreate(opts types.VolumeCreateOptions) (*types.Volume, error) {
	path, err := namespacedPath(opts.Namespace, VolumeAPIPrefix)
	if err != nil {
		return nil, err
	}
	resp, err := c.do("POST", path, doOptions{
		data:    opts,
		context: opts.Context,
	})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var volume types.Volume
	if err := json.NewDecoder(resp.Body).Decode(&volume); err != nil {
		return nil, err
	}
	return &volume, nil
}

// VolumeUpdate updates a volume on the server.
func (c *Client) VolumeUpdate(opts types.VolumeUpdateOptions) (*types.Volume, error) {
	ref := opts.Name
	if IsUUID(opts.ID) {
		ref = opts.ID
	}
	path, err := namespacedRefPath(opts.Namespace, VolumeAPIPrefix, ref)
	if err != nil {
		return nil, err
	}
	resp, err := c.do("PUT", path, doOptions{
		data:    opts,
		context: opts.Context,
	})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var volume types.Volume
	if err := json.NewDecoder(resp.Body).Decode(&volume); err != nil {
		return nil, err
	}
	return &volume, nil
}

// VolumeDelete removes a volume by its reference.
func (c *Client) VolumeDelete(opts types.DeleteOptions) error {
	deleteOpts := doOptions{
		namespace: opts.Namespace,
		force:     opts.Force,
		context:   opts.Context,
	}
	resp, err := c.do("DELETE", VolumeAPIPrefix+"/"+opts.Name, deleteOpts)
	if err != nil {
		if e, ok := err.(*Error); ok {
			if e.Status == http.StatusNotFound {
				return ErrNoSuchVolume
			}
			if e.Status == http.StatusConflict {
				return ErrVolumeInUse
			}
		}
		return err
	}
	defer resp.Body.Close()
	return nil
}

// VolumeMount updates the volume with the client that mounted it.
func (c *Client) VolumeMount(opts types.VolumeMountOptions) error {
	ref := opts.Name
	if IsUUID(opts.ID) {
		ref = opts.ID
	}
	path, err := namespacedRefPath(opts.Namespace, VolumeAPIPrefix, ref)
	if err != nil {
		return err
	}
	resp, err := c.do("POST", path+"/mount", doOptions{
		data:    opts,
		context: opts.Context,
	})
	if err != nil {
		if e, ok := err.(*Error); ok {
			if e.Status == http.StatusNotFound {
				return ErrNoSuchVolume
			}
			if e.Status == http.StatusConflict {
				return ErrVolumeInUse
			}
		}
		return err
	}
	defer resp.Body.Close()
	return nil
}

// VolumeUnmount removes the client from the mount reference.
func (c *Client) VolumeUnmount(opts types.VolumeUnmountOptions) error {
	ref := opts.Name
	if IsUUID(opts.ID) {
		ref = opts.ID
	}
	path, err := namespacedRefPath(opts.Namespace, VolumeAPIPrefix, ref)
	if err != nil {
		return err
	}
	resp, err := c.do("POST", path+"/unmount", doOptions{
		data:    opts,
		context: opts.Context,
	})
	if err != nil {
		if e, ok := err.(*Error); ok {
			if e.Status == http.StatusNotFound {
				return ErrNoSuchVolume
			}
			if e.Status == http.StatusConflict {
				return ErrVolumeInUse
			}
		}
		return err
	}
	defer resp.Body.Close()
	return nil
}
