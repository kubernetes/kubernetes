package volumes

import (
	"fmt"

	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/blockstorage/v1/volumes"
	"github.com/rackspace/gophercloud/pagination"
)

type CreateOpts struct {
	os.CreateOpts
}

func (opts CreateOpts) ToVolumeCreateMap() (map[string]interface{}, error) {
	if opts.Size < 75 || opts.Size > 1024 {
		return nil, fmt.Errorf("Size field must be between 75 and 1024")
	}

	return opts.CreateOpts.ToVolumeCreateMap()
}

// Create will create a new Volume based on the values in CreateOpts. To extract
// the Volume object from the response, call the Extract method on the
// CreateResult.
func Create(client *gophercloud.ServiceClient, opts os.CreateOptsBuilder) CreateResult {
	return CreateResult{os.Create(client, opts)}
}

// Delete will delete the existing Volume with the provided ID.
func Delete(client *gophercloud.ServiceClient, id string) os.DeleteResult {
	return os.Delete(client, id)
}

// Get retrieves the Volume with the provided ID. To extract the Volume object
// from the response, call the Extract method on the GetResult.
func Get(client *gophercloud.ServiceClient, id string) GetResult {
	return GetResult{os.Get(client, id)}
}

// List returns volumes optionally limited by the conditions provided in ListOpts.
func List(client *gophercloud.ServiceClient) pagination.Pager {
	return os.List(client, os.ListOpts{})
}

// UpdateOpts contain options for updating an existing Volume. This object is passed
// to the volumes.Update function. For more information about the parameters, see
// the Volume object.
type UpdateOpts struct {
	// OPTIONAL
	Name string
	// OPTIONAL
	Description string
}

// ToVolumeUpdateMap assembles a request body based on the contents of an
// UpdateOpts.
func (opts UpdateOpts) ToVolumeUpdateMap() (map[string]interface{}, error) {
	v := make(map[string]interface{})

	if opts.Description != "" {
		v["display_description"] = opts.Description
	}
	if opts.Name != "" {
		v["display_name"] = opts.Name
	}

	return map[string]interface{}{"volume": v}, nil
}

// Update will update the Volume with provided information. To extract the updated
// Volume from the response, call the Extract method on the UpdateResult.
func Update(client *gophercloud.ServiceClient, id string, opts os.UpdateOptsBuilder) UpdateResult {
	return UpdateResult{os.Update(client, id, opts)}
}
