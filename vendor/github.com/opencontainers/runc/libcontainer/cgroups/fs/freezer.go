// +build linux

package fs

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
	"golang.org/x/sys/unix"
)

type FreezerGroup struct {
}

func (s *FreezerGroup) Name() string {
	return "freezer"
}

func (s *FreezerGroup) Apply(path string, d *cgroupData) error {
	return join(path, d.pid)
}

func (s *FreezerGroup) Set(path string, cgroup *configs.Cgroup) error {
	switch cgroup.Resources.Freezer {
	case configs.Frozen:
		// As per older kernel docs (freezer-subsystem.txt before
		// kernel commit ef9fe980c6fcc1821), if FREEZING is seen,
		// userspace should either retry or thaw. While current
		// kernel cgroup v1 docs no longer mention a need to retry,
		// the kernel (tested on v5.4, Ubuntu 20.04) can't reliably
		// freeze a cgroup while new processes keep appearing in it
		// (either via fork/clone or by writing new PIDs to
		// cgroup.procs).
		//
		// The number of retries below is chosen to have a decent
		// chance to succeed even in the worst case scenario (runc
		// pause/unpause with parallel runc exec).
		//
		// Adding any amount of sleep in between retries did not
		// increase the chances of successful freeze.
		for i := 0; i < 1000; i++ {
			if err := fscommon.WriteFile(path, "freezer.state", string(configs.Frozen)); err != nil {
				return err
			}

			state, err := fscommon.ReadFile(path, "freezer.state")
			if err != nil {
				return err
			}
			state = strings.TrimSpace(state)
			switch state {
			case "FREEZING":
				continue
			case string(configs.Frozen):
				return nil
			default:
				// should never happen
				return fmt.Errorf("unexpected state %s while freezing", strings.TrimSpace(state))
			}
		}
		// Despite our best efforts, it got stuck in FREEZING.
		// Leaving it in this state is bad and dangerous, so
		// let's (try to) thaw it back and error out.
		_ = fscommon.WriteFile(path, "freezer.state", string(configs.Thawed))
		return errors.New("unable to freeze")
	case configs.Thawed:
		return fscommon.WriteFile(path, "freezer.state", string(configs.Thawed))
	case configs.Undefined:
		return nil
	default:
		return fmt.Errorf("Invalid argument '%s' to freezer.state", string(cgroup.Resources.Freezer))
	}
}

func (s *FreezerGroup) GetStats(path string, stats *cgroups.Stats) error {
	return nil
}

func (s *FreezerGroup) GetState(path string) (configs.FreezerState, error) {
	for {
		state, err := fscommon.ReadFile(path, "freezer.state")
		if err != nil {
			// If the kernel is too old, then we just treat the freezer as
			// being in an "undefined" state.
			if os.IsNotExist(err) || errors.Is(err, unix.ENODEV) {
				err = nil
			}
			return configs.Undefined, err
		}
		switch strings.TrimSpace(state) {
		case "THAWED":
			return configs.Thawed, nil
		case "FROZEN":
			return configs.Frozen, nil
		case "FREEZING":
			// Make sure we get a stable freezer state, so retry if the cgroup
			// is still undergoing freezing. This should be a temporary delay.
			time.Sleep(1 * time.Millisecond)
			continue
		default:
			return configs.Undefined, fmt.Errorf("unknown freezer.state %q", state)
		}
	}
}
