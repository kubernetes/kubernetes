// +build !windows

package windows

import (
	"fmt"

	"github.com/docker/docker/daemon/execdriver"
)

func NewDriver(root, initPath string) (execdriver.Driver, error) {
	return nil, fmt.Errorf("Windows driver not supported on non-Windows")
}
