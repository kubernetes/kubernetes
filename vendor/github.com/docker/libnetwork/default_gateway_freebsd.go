package libnetwork

import "github.com/docker/libnetwork/types"

const libnGWNetwork = "docker_gwbridge"

func getPlatformOption() EndpointOption {
	return nil
}

func (c *controller) createGWNetwork() (Network, error) {
	return nil, types.NotImplementedErrorf("default gateway functionality is not implemented in freebsd")
}
