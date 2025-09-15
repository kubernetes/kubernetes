package fs

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/opencontainers/cgroups"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

type FreezerGroup struct{}

func (s *FreezerGroup) Name() string {
	return "freezer"
}

func (s *FreezerGroup) Apply(path string, _ *cgroups.Resources, pid int) error {
	return apply(path, pid)
}

func (s *FreezerGroup) Set(path string, r *cgroups.Resources) (Err error) {
	switch r.Freezer {
	case cgroups.Frozen:
		defer func() {
			if Err != nil {
				// Freezing failed, and it is bad and dangerous
				// to leave the cgroup in FROZEN or FREEZING
				// state, so (try to) thaw it back.
				_ = cgroups.WriteFile(path, "freezer.state", string(cgroups.Thawed))
			}
		}()

		// As per older kernel docs (freezer-subsystem.txt before
		// kernel commit ef9fe980c6fcc1821), if FREEZING is seen,
		// userspace should either retry or thaw. While current
		// kernel cgroup v1 docs no longer mention a need to retry,
		// even a recent kernel (v5.4, Ubuntu 20.04) can't reliably
		// freeze a cgroup v1 while new processes keep appearing in it
		// (either via fork/clone or by writing new PIDs to
		// cgroup.procs).
		//
		// The numbers below are empirically chosen to have a decent
		// chance to succeed in various scenarios ("runc pause/unpause
		// with parallel runc exec" and "bare freeze/unfreeze on a very
		// slow system"), tested on RHEL7 and Ubuntu 20.04 kernels.
		//
		// Adding any amount of sleep in between retries did not
		// increase the chances of successful freeze in "pause/unpause
		// with parallel exec" reproducer. OTOH, adding an occasional
		// sleep helped for the case where the system is extremely slow
		// (CentOS 7 VM on GHA CI).
		//
		// Alas, this is still a game of chances, since the real fix
		// belong to the kernel (cgroup v2 do not have this bug).

		for i := 0; i < 1000; i++ {
			if i%50 == 49 {
				// Occasional thaw and sleep improves
				// the chances to succeed in freezing
				// in case new processes keep appearing
				// in the cgroup.
				_ = cgroups.WriteFile(path, "freezer.state", string(cgroups.Thawed))
				time.Sleep(10 * time.Millisecond)
			}

			if err := cgroups.WriteFile(path, "freezer.state", string(cgroups.Frozen)); err != nil {
				return err
			}

			if i%25 == 24 {
				// Occasional short sleep before reading
				// the state back also improves the chances to
				// succeed in freezing in case of a very slow
				// system.
				time.Sleep(10 * time.Microsecond)
			}
			state, err := cgroups.ReadFile(path, "freezer.state")
			if err != nil {
				return err
			}
			state = strings.TrimSpace(state)
			switch state {
			case "FREEZING":
				continue
			case string(cgroups.Frozen):
				if i > 1 {
					logrus.Debugf("frozen after %d retries", i)
				}
				return nil
			default:
				// should never happen
				return fmt.Errorf("unexpected state %s while freezing", strings.TrimSpace(state))
			}
		}
		// Despite our best efforts, it got stuck in FREEZING.
		return errors.New("unable to freeze")
	case cgroups.Thawed:
		return cgroups.WriteFile(path, "freezer.state", string(cgroups.Thawed))
	case cgroups.Undefined:
		return nil
	default:
		return fmt.Errorf("Invalid argument '%s' to freezer.state", string(r.Freezer))
	}
}

func (s *FreezerGroup) GetStats(path string, stats *cgroups.Stats) error {
	return nil
}

func (s *FreezerGroup) GetState(path string) (cgroups.FreezerState, error) {
	for {
		state, err := cgroups.ReadFile(path, "freezer.state")
		if err != nil {
			// If the kernel is too old, then we just treat the freezer as
			// being in an "undefined" state.
			if os.IsNotExist(err) || errors.Is(err, unix.ENODEV) {
				err = nil
			}
			return cgroups.Undefined, err
		}
		switch strings.TrimSpace(state) {
		case "THAWED":
			return cgroups.Thawed, nil
		case "FROZEN":
			// Find out whether the cgroup is frozen directly,
			// or indirectly via an ancestor.
			self, err := cgroups.ReadFile(path, "freezer.self_freezing")
			if err != nil {
				// If the kernel is too old, then we just treat
				// it as being frozen.
				if errors.Is(err, os.ErrNotExist) || errors.Is(err, unix.ENODEV) {
					err = nil
				}
				return cgroups.Frozen, err
			}
			switch self {
			case "0\n":
				return cgroups.Thawed, nil
			case "1\n":
				return cgroups.Frozen, nil
			default:
				return cgroups.Undefined, fmt.Errorf(`unknown "freezer.self_freezing" state: %q`, self)
			}
		case "FREEZING":
			// Make sure we get a stable freezer state, so retry if the cgroup
			// is still undergoing freezing. This should be a temporary delay.
			time.Sleep(1 * time.Millisecond)
			continue
		default:
			return cgroups.Undefined, fmt.Errorf("unknown freezer.state %q", state)
		}
	}
}
