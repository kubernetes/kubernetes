// +build !exclude_graphdriver_devicemapper,linux

package daemon

import (
	_ "github.com/docker/docker/daemon/graphdriver/devmapper"
)
