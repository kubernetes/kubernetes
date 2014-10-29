package volumetypes

import (
	"github.com/racker/perigee"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToVolumeTypeCreateMap() (map[string]interface{}, error)
}

// CreateOpts are options for creating a volume type.
type CreateOpts struct {
	// OPTIONAL. See VolumeType.
	ExtraSpecs map[string]interface{}
	// OPTIONAL. See VolumeType.
	Name string
}

// ToVolumeTypeCreateMap casts a CreateOpts struct to a map.
func (opts CreateOpts) ToVolumeTypeCreateMap() (map[string]interface{}, error) {
	vt := make(map[string]interface{})

	if opts.ExtraSpecs != nil {
		vt["extra_specs"] = opts.ExtraSpecs
	}
	if opts.Name != "" {
		vt["name"] = opts.Name
	}

	return map[string]interface{}{"volume_type": vt}, nil
}

// Create will create a new volume. To extract the created volume type object,
// call the Extract method on the CreateResult.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToVolumeTypeCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = perigee.Request("POST", createURL(client), perigee.Options{
		MoreHeaders: client.AuthenticatedHeaders(),
		OkCodes:     []int{200, 201},
		ReqBody:     &reqBody,
		Results:     &res.Body,
	})
	return res
}

// Delete will delete the volume type with the provided ID.
func Delete(client *gophercloud.ServiceClient, id string) DeleteResult {
	var res DeleteResult
	_, res.Err = perigee.Request("DELETE", deleteURL(client, id), perigee.Options{
		MoreHeaders: client.AuthenticatedHeaders(),
		OkCodes:     []int{202},
	})
	return res
}

// Get will retrieve the volume type with the provided ID. To extract the volume
// type from the result, call the Extract method on the GetResult.
func Get(client *gophercloud.ServiceClient, id string) GetResult {
	var res GetResult
	_, err := perigee.Request("GET", getURL(client, id), perigee.Options{
		MoreHeaders: client.AuthenticatedHeaders(),
		OkCodes:     []int{200},
		Results:     &res.Body,
	})
	res.Err = err
	return res
}

// List returns all volume types.
func List(client *gophercloud.ServiceClient) pagination.Pager {
	createPage := func(r pagination.PageResult) pagination.Page {
		return ListResult{pagination.SinglePageBase(r)}
	}

	return pagination.NewPager(client, listURL(client), createPage)
}
