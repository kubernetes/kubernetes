package daemon

import (
	"github.com/docker/docker/container"
	"github.com/docker/docker/libcontainerd"
)

// postRunProcessing perfoms any processing needed on the container after it has stopped.
func (daemon *Daemon) postRunProcessing(_ *container.Container, _ libcontainerd.EventInfo) error {
	return nil
}
