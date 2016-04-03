// +build windows

package windows

import (
	"fmt"

	"github.com/docker/docker/daemon/execdriver"
)

func (d *driver) Pause(c *execdriver.Command) error {
	return fmt.Errorf("Windows: Containers cannot be paused")
}

func (d *driver) Unpause(c *execdriver.Command) error {
	return fmt.Errorf("Windows: Containers cannot be paused")
}
