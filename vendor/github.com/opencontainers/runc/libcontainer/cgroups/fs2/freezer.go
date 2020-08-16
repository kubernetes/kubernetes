// +build linux

package fs2

import (
	stdErrors "errors"
	"os"
	"strings"

	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/pkg/errors"
	"golang.org/x/sys/unix"
)

func setFreezer(dirPath string, state configs.FreezerState) error {
	if err := supportsFreezer(dirPath); err != nil {
		// We can ignore this request as long as the user didn't ask us to
		// freeze the container (since without the freezer cgroup, that's a
		// no-op).
		if state == configs.Undefined || state == configs.Thawed {
			err = nil
		}
		return errors.Wrap(err, "freezer not supported")
	}

	var stateStr string
	switch state {
	case configs.Undefined:
		return nil
	case configs.Frozen:
		stateStr = "1"
	case configs.Thawed:
		stateStr = "0"
	default:
		return errors.Errorf("invalid freezer state %q requested", state)
	}

	if err := fscommon.WriteFile(dirPath, "cgroup.freeze", stateStr); err != nil {
		return err
	}
	// Confirm that the cgroup did actually change states.
	if actualState, err := getFreezer(dirPath); err != nil {
		return err
	} else if actualState != state {
		return errors.Errorf(`expected "cgroup.freeze" to be in state %q but was in %q`, state, actualState)
	}
	return nil
}

func supportsFreezer(dirPath string) error {
	_, err := fscommon.ReadFile(dirPath, "cgroup.freeze")
	return err
}

func getFreezer(dirPath string) (configs.FreezerState, error) {
	state, err := fscommon.ReadFile(dirPath, "cgroup.freeze")
	if err != nil {
		// If the kernel is too old, then we just treat the freezer as being in
		// an "undefined" state.
		if os.IsNotExist(err) || stdErrors.Is(err, unix.ENODEV) {
			err = nil
		}
		return configs.Undefined, err
	}
	switch strings.TrimSpace(state) {
	case "0":
		return configs.Thawed, nil
	case "1":
		return configs.Frozen, nil
	default:
		return configs.Undefined, errors.Errorf(`unknown "cgroup.freeze" state: %q`, state)
	}
}
