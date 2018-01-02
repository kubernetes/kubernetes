// +build !have_chainfs

package chainfs

import (
	"errors"

	"github.com/docker/docker/daemon/graphdriver"
	"github.com/docker/docker/pkg/idtools"
	"github.com/libopenstorage/openstorage/api"
)

const (
	// Name of the driver
	Name = "chainfs"
	// Type of the driver
	Type = api.DriverType_DRIVER_TYPE_GRAPH
)

var (
	errUnsupported = errors.New("chainfs not supported on this platform")
)

// Init initializes the graphdriver
func Init(home string, options []string, uidMaps, gidMaps []idtools.IDMap) (graphdriver.Driver, error) {
	return nil, errUnsupported
}
