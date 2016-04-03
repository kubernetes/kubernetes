// +build windows

package daemon

import (
	"os"

	"github.com/docker/docker/daemon/execdriver"
)

// Not supported on Windows
func copyOwnership(source, destination string) error {
	return nil
}

func (container *Container) setupMounts() ([]execdriver.Mount, error) {
	return nil, nil
}

func migrateVolume(id, vfs string) error {
	return nil
}

func validVolumeLayout(files []os.FileInfo) bool {
	return true
}
