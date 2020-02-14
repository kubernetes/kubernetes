// +build linux

package fs2

import (
	"strconv"
	"strings"

	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/pkg/errors"
)

func setFreezer(dirPath string, state configs.FreezerState) error {
	var desired int
	switch state {
	case configs.Undefined:
		return nil
	case configs.Frozen:
		desired = 1
	case configs.Thawed:
		desired = 0
	default:
		return errors.Errorf("unknown freezer state %+v", state)
	}
	supportedErr := supportsFreezer(dirPath)
	if supportedErr != nil && desired != 0 {
		// can ignore error if desired == 1
		return errors.Wrap(supportedErr, "freezer not supported")
	}
	return freezeWithInt(dirPath, desired)
}

func supportsFreezer(dirPath string) error {
	_, err := fscommon.ReadFile(dirPath, "cgroup.freeze")
	return err
}

// freeze writes desired int to "cgroup.freeze".
func freezeWithInt(dirPath string, desired int) error {
	desiredS := strconv.Itoa(desired)
	if err := fscommon.WriteFile(dirPath, "cgroup.freeze", desiredS); err != nil {
		return err
	}
	got, err := fscommon.ReadFile(dirPath, "cgroup.freeze")
	if err != nil {
		return err
	}
	if gotS := strings.TrimSpace(string(got)); gotS != desiredS {
		return errors.Errorf("expected \"cgroup.freeze\" in %q to be %q, got %q", dirPath, desiredS, gotS)
	}
	return nil
}
