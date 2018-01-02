//+build windows

package daemon

import (
	"github.com/docker/docker/container"
)

func (daemon *Daemon) saveApparmorConfig(container *container.Container) error {
	return nil
}
