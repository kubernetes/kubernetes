package volumetypes

import (
	"github.com/mitchellh/mapstructure"
	os "github.com/rackspace/gophercloud/openstack/blockstorage/v1/volumetypes"
	"github.com/rackspace/gophercloud/pagination"
)

type VolumeType os.VolumeType

type GetResult struct {
	os.GetResult
}

// Extract will get the Volume Type struct out of the response.
func (r GetResult) Extract() (*VolumeType, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		VolumeType *VolumeType `json:"volume_type" mapstructure:"volume_type"`
	}

	err := mapstructure.Decode(r.Body, &res)

	return res.VolumeType, err
}

func ExtractVolumeTypes(page pagination.Page) ([]VolumeType, error) {
	var response struct {
		VolumeTypes []VolumeType `mapstructure:"volume_types"`
	}

	err := mapstructure.Decode(page.(os.ListResult).Body, &response)
	return response.VolumeTypes, err
}
