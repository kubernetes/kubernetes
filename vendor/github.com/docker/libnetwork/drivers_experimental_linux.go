package libnetwork

import "github.com/docker/libnetwork/drivers/ipvlan"

func additionalDrivers() []initializer {
	return []initializer{
		{ipvlan.Init, "ipvlan"},
	}
}
