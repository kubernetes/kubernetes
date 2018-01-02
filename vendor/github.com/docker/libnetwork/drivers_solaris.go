package libnetwork

import (
	"github.com/docker/libnetwork/drivers/null"
	"github.com/docker/libnetwork/drivers/solaris/bridge"
	"github.com/docker/libnetwork/drivers/solaris/overlay"
)

func getInitializers(experimental bool) []initializer {
	return []initializer{
		{overlay.Init, "overlay"},
		{bridge.Init, "bridge"},
		{null.Init, "null"},
	}
}
