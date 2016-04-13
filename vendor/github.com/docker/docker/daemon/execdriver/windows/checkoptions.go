// +build windows

package windows

import (
	"errors"

	"github.com/docker/docker/daemon/execdriver"
)

func checkSupportedOptions(c *execdriver.Command) error {
	// Windows doesn't support read-only root filesystem
	if c.ReadonlyRootfs {
		return errors.New("Windows does not support the read-only root filesystem option")
	}

	// Windows doesn't support username
	if c.ProcessConfig.User != "" {
		return errors.New("Windows does not support the username option")
	}

	// Windows doesn't support custom lxc options
	if c.LxcConfig != nil {
		return errors.New("Windows does not support lxc options")
	}

	// Windows doesn't support ulimit
	if c.Resources.Rlimits != nil {
		return errors.New("Windows does not support ulimit options")
	}

	// TODO Windows: Validate other fields which Windows doesn't support, factor
	// out where applicable per platform.

	return nil
}
