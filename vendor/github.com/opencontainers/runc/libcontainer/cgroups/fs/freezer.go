// +build linux

package fs

import (
	"fmt"
	"strings"
	"time"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
)

type FreezerGroup struct {
}

func (s *FreezerGroup) Name() string {
	return "freezer"
}

func (s *FreezerGroup) Apply(d *cgroupData) error {
	dir, err := d.join("freezer")
	if err != nil && !cgroups.IsNotFound(err) {
		return err
	}

	if err := s.Set(dir, d.config); err != nil {
		return err
	}

	return nil
}

func (s *FreezerGroup) Set(path string, cgroup *configs.Cgroup) error {
	switch cgroup.Freezer {
	case configs.Frozen, configs.Thawed:
		if err := writeFile(path, "freezer.state", string(cgroup.Freezer)); err != nil {
			return err
		}

		for {
			state, err := readFile(path, "freezer.state")
			if err != nil {
				return err
			}
			if strings.TrimSpace(state) == string(cgroup.Freezer) {
				break
			}
			time.Sleep(1 * time.Millisecond)
		}
	case configs.Undefined:
		return nil
	default:
		return fmt.Errorf("Invalid argument '%s' to freezer.state", string(cgroup.Freezer))
	}

	return nil
}

func (s *FreezerGroup) Remove(d *cgroupData) error {
	return removePath(d.path("freezer"))
}

func (s *FreezerGroup) GetStats(path string, stats *cgroups.Stats) error {
	return nil
}
