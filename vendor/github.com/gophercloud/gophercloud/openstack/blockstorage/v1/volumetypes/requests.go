package volumetypes

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToVolumeTypeCreateMap() (map[string]interface{}, error)
}

// CreateOpts are options for creating a volume type.
type CreateOpts struct {
	// See VolumeType.
	ExtraSpecs map[string]interface{} `json:"extra_specs,omitempty"`
	// See VolumeType.
	Name string `json:"name,omitempty"`
}

// ToVolumeTypeCreateMap casts a CreateOpts struct to a map.
func (opts CreateOpts) ToVolumeTypeCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "volume_type")
}

// Create will create a new volume. To extract the created volume type object,
// call the Extract method on the CreateResult.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToVolumeTypeCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(createURL(client), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201},
	})
	return
}

// Delete will delete the volume type with the provided ID.
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, id), nil)
	return
}

// Get will retrieve the volume type with the provided ID. To extract the volume
// type from the result, call the Extract method on the GetResult.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body, nil)
	return
}

// List returns all volume types.
func List(client *gophercloud.ServiceClient) pagination.Pager {
	return pagination.NewPager(client, listURL(client), func(r pagination.PageResult) pagination.Page {
		return VolumeTypePage{pagination.SinglePageBase(r)}
	})
}
