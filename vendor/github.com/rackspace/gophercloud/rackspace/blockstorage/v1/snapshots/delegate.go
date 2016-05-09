package snapshots

import (
	"errors"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"

	os "github.com/rackspace/gophercloud/openstack/blockstorage/v1/snapshots"
)

func updateURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("snapshots", id)
}

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToSnapshotCreateMap() (map[string]interface{}, error)
}

// CreateOpts contains options for creating a Snapshot. This object is passed to
// the snapshots.Create function. For more information about these parameters,
// see the Snapshot object.
type CreateOpts struct {
	// REQUIRED
	VolumeID string
	// OPTIONAL
	Description string
	// OPTIONAL
	Force bool
	// OPTIONAL
	Name string
}

// ToSnapshotCreateMap assembles a request body based on the contents of a
// CreateOpts.
func (opts CreateOpts) ToSnapshotCreateMap() (map[string]interface{}, error) {
	s := make(map[string]interface{})

	if opts.VolumeID == "" {
		return nil, errors.New("Required CreateOpts field 'VolumeID' not set.")
	}

	s["volume_id"] = opts.VolumeID

	if opts.Description != "" {
		s["display_description"] = opts.Description
	}
	if opts.Name != "" {
		s["display_name"] = opts.Name
	}
	if opts.Force {
		s["force"] = opts.Force
	}

	return map[string]interface{}{"snapshot": s}, nil
}

// Create will create a new Snapshot based on the values in CreateOpts. To
// extract the Snapshot object from the response, call the Extract method on the
// CreateResult.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) CreateResult {
	return CreateResult{os.Create(client, opts)}
}

// Delete will delete the existing Snapshot with the provided ID.
func Delete(client *gophercloud.ServiceClient, id string) os.DeleteResult {
	return os.Delete(client, id)
}

// Get retrieves the Snapshot with the provided ID. To extract the Snapshot
// object from the response, call the Extract method on the GetResult.
func Get(client *gophercloud.ServiceClient, id string) GetResult {
	return GetResult{os.Get(client, id)}
}

// List returns Snapshots.
func List(client *gophercloud.ServiceClient) pagination.Pager {
	return os.List(client, os.ListOpts{})
}

// UpdateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the main Update operation in this package. Since many
// extensions decorate or modify the common logic, it is useful for them to
// satisfy a basic interface in order for them to be used.
type UpdateOptsBuilder interface {
	ToSnapshotUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts is the common options struct used in this package's Update
// operation.
type UpdateOpts struct {
	Name        string
	Description string
}

// ToSnapshotUpdateMap casts a UpdateOpts struct to a map.
func (opts UpdateOpts) ToSnapshotUpdateMap() (map[string]interface{}, error) {
	s := make(map[string]interface{})

	if opts.Name != "" {
		s["display_name"] = opts.Name
	}
	if opts.Description != "" {
		s["display_description"] = opts.Description
	}

	return map[string]interface{}{"snapshot": s}, nil
}

// Update accepts a UpdateOpts struct and updates an existing snapshot using the
// values provided.
func Update(c *gophercloud.ServiceClient, snapshotID string, opts UpdateOptsBuilder) UpdateResult {
	var res UpdateResult

	reqBody, err := opts.ToSnapshotUpdateMap()
	if err != nil {
		res.Err = err
		return res
	}

	// Send request to API
	_, res.Err = c.Request("PUT", updateURL(c, snapshotID), gophercloud.RequestOpts{
		JSONBody:     &reqBody,
		JSONResponse: &res.Body,
		OkCodes:      []int{200, 201},
	})

	return res
}
