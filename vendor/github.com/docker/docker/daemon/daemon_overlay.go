// +build !exclude_graphdriver_overlay,linux

package daemon

import (
	_ "github.com/docker/docker/daemon/graphdriver/overlay"
)
