package volumes

import (
	os "github.com/rackspace/gophercloud/openstack/blockstorage/v1/volumes"
	"github.com/rackspace/gophercloud/pagination"

	"github.com/mitchellh/mapstructure"
)

// Volume wraps an Openstack volume
type Volume os.Volume

// CreateResult represents the result of a create operation
type CreateResult struct {
	os.CreateResult
}

// GetResult represents the result of a get operation
type GetResult struct {
	os.GetResult
}

// UpdateResult represents the result of an update operation
type UpdateResult struct {
	os.UpdateResult
}

func commonExtract(resp interface{}, err error) (*Volume, error) {
	if err != nil {
		return nil, err
	}

	var respStruct struct {
		Volume *Volume `json:"volume"`
	}

	err = mapstructure.Decode(resp, &respStruct)

	return respStruct.Volume, err
}

// Extract will get the Volume object out of the GetResult object.
func (r GetResult) Extract() (*Volume, error) {
	return commonExtract(r.Body, r.Err)
}

// Extract will get the Volume object out of the CreateResult object.
func (r CreateResult) Extract() (*Volume, error) {
	return commonExtract(r.Body, r.Err)
}

// Extract will get the Volume object out of the UpdateResult object.
func (r UpdateResult) Extract() (*Volume, error) {
	return commonExtract(r.Body, r.Err)
}

// ExtractVolumes extracts and returns Volumes. It is used while iterating over a volumes.List call.
func ExtractVolumes(page pagination.Page) ([]Volume, error) {
	var response struct {
		Volumes []Volume `json:"volumes"`
	}

	err := mapstructure.Decode(page.(os.ListResult).Body, &response)

	return response.Volumes, err
}
