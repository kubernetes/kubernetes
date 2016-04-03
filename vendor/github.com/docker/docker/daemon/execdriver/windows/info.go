// +build windows

package windows

import "github.com/docker/docker/daemon/execdriver"

type info struct {
	ID     string
	driver *driver
}

func (d *driver) Info(id string) execdriver.Info {
	return &info{
		ID:     id,
		driver: d,
	}
}

func (i *info) IsRunning() bool {
	var running bool
	running = true // TODO Need an HCS API
	return running
}
