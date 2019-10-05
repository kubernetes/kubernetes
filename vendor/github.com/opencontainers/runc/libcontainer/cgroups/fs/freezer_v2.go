// +build linux

package fs

import (
	"fmt"
	"strings"
	"time"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
)

type FreezerGroupV2 struct {
}

func (s *FreezerGroupV2) Name() string {
	return "freezer"
}

func (s *FreezerGroupV2) Apply(d *cgroupData) error {
	_, err := d.join("freezer")
	if err != nil && !cgroups.IsNotFound(err) {
		return err
	}
	return nil
}

func (s *FreezerGroupV2) Set(path string, cgroup *configs.Cgroup) error {
	var desiredState string
	filename := "cgroup.freeze"
	if cgroup.Resources.Freezer == configs.Frozen {
		desiredState = "1"
	} else {
		desiredState = "0"
	}

	switch cgroup.Resources.Freezer {
	case configs.Frozen, configs.Thawed:
		for {
			// In case this loop does not exit because it doesn't get the expected
			// state, let's write again this state, hoping it's going to be properly
			// set this time. Otherwise, this loop could run infinitely, waiting for
			// a state change that would never happen.
			if err := writeFile(path, filename, desiredState); err != nil {
				return err
			}

			state, err := readFile(path, filename)
			if err != nil {
				return err
			}
			if strings.TrimSpace(state) == desiredState {
				break
			}

			time.Sleep(1 * time.Millisecond)
		}
	case configs.Undefined:
		return nil
	default:
		return fmt.Errorf("Invalid argument '%s' to freezer.state", string(cgroup.Resources.Freezer))
	}

	return nil
}

func (s *FreezerGroupV2) Remove(d *cgroupData) error {
	return removePath(d.path("freezer"))
}

func (s *FreezerGroupV2) GetStats(path string, stats *cgroups.Stats) error {
	return nil
}
