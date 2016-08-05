package diskconfig

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack/compute/v2/servers"
	"github.com/rackspace/gophercloud/pagination"
)

func commonExtract(result gophercloud.Result) (*DiskConfig, error) {
	var resp struct {
		Server struct {
			DiskConfig string `mapstructure:"OS-DCF:diskConfig"`
		} `mapstructure:"server"`
	}

	err := mapstructure.Decode(result.Body, &resp)
	if err != nil {
		return nil, err
	}

	config := DiskConfig(resp.Server.DiskConfig)
	return &config, nil
}

// ExtractGet returns the disk configuration from a servers.Get call.
func ExtractGet(result servers.GetResult) (*DiskConfig, error) {
	return commonExtract(result.Result)
}

// ExtractUpdate returns the disk configuration from a servers.Update call.
func ExtractUpdate(result servers.UpdateResult) (*DiskConfig, error) {
	return commonExtract(result.Result)
}

// ExtractRebuild returns the disk configuration from a servers.Rebuild call.
func ExtractRebuild(result servers.RebuildResult) (*DiskConfig, error) {
	return commonExtract(result.Result)
}

// ExtractDiskConfig returns the DiskConfig setting for a specific server acquired from an
// servers.ExtractServers call, while iterating through a Pager.
func ExtractDiskConfig(page pagination.Page, index int) (*DiskConfig, error) {
	casted := page.(servers.ServerPage).Body

	type server struct {
		DiskConfig string `mapstructure:"OS-DCF:diskConfig"`
	}
	var response struct {
		Servers []server `mapstructure:"servers"`
	}

	err := mapstructure.Decode(casted, &response)
	if err != nil {
		return nil, err
	}

	config := DiskConfig(response.Servers[index].DiskConfig)
	return &config, nil
}
