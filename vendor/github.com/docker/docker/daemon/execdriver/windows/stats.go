// +build windows

package windows

import (
	"fmt"

	"github.com/docker/docker/daemon/execdriver"
)

func (d *driver) Stats(id string) (*execdriver.ResourceStats, error) {
	return nil, fmt.Errorf("Windows: Stats not implemented")
}
