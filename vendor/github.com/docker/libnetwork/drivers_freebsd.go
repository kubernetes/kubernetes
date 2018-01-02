package libnetwork

import (
	"github.com/docker/libnetwork/drivers/null"
	"github.com/docker/libnetwork/drivers/remote"
)

func getInitializers(experimental bool) []initializer {
	return []initializer{
		{null.Init, "null"},
		{remote.Init, "remote"},
	}
}
