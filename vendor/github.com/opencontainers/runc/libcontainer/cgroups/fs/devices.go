// +build linux

package fs

import (
	"bytes"
	"errors"
	"reflect"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/devices"
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runc/libcontainer/system"
)

type DevicesGroup struct {
	testingSkipFinalCheck bool
}

func (s *DevicesGroup) Name() string {
	return "devices"
}

func (s *DevicesGroup) Apply(path string, d *cgroupData) error {
	if d.config.SkipDevices {
		return nil
	}
	if path == "" {
		// Return error here, since devices cgroup
		// is a hard requirement for container's security.
		return errSubsystemDoesNotExist
	}
	return join(path, d.pid)
}

func loadEmulator(path string) (*devices.Emulator, error) {
	list, err := fscommon.ReadFile(path, "devices.list")
	if err != nil {
		return nil, err
	}
	return devices.EmulatorFromList(bytes.NewBufferString(list))
}

func buildEmulator(rules []*configs.DeviceRule) (*devices.Emulator, error) {
	// This defaults to a white-list -- which is what we want!
	emu := &devices.Emulator{}
	for _, rule := range rules {
		if err := emu.Apply(*rule); err != nil {
			return nil, err
		}
	}
	return emu, nil
}

func (s *DevicesGroup) Set(path string, cgroup *configs.Cgroup) error {
	if system.RunningInUserNS() || cgroup.SkipDevices {
		return nil
	}

	// Generate two emulators, one for the current state of the cgroup and one
	// for the requested state by the user.
	current, err := loadEmulator(path)
	if err != nil {
		return err
	}
	target, err := buildEmulator(cgroup.Resources.Devices)
	if err != nil {
		return err
	}

	// Compute the minimal set of transition rules needed to achieve the
	// requested state.
	transitionRules, err := current.Transition(target)
	if err != nil {
		return err
	}
	for _, rule := range transitionRules {
		file := "devices.deny"
		if rule.Allow {
			file = "devices.allow"
		}
		if err := fscommon.WriteFile(path, file, rule.CgroupString()); err != nil {
			return err
		}
	}

	// Final safety check -- ensure that the resulting state is what was
	// requested. This is only really correct for white-lists, but for
	// black-lists we can at least check that the cgroup is in the right mode.
	//
	// This safety-check is skipped for the unit tests because we cannot
	// currently mock devices.list correctly.
	if !s.testingSkipFinalCheck {
		currentAfter, err := loadEmulator(path)
		if err != nil {
			return err
		}
		if !target.IsBlacklist() && !reflect.DeepEqual(currentAfter, target) {
			return errors.New("resulting devices cgroup doesn't precisely match target")
		} else if target.IsBlacklist() != currentAfter.IsBlacklist() {
			return errors.New("resulting devices cgroup doesn't match target mode")
		}
	}
	return nil
}

func (s *DevicesGroup) GetStats(path string, stats *cgroups.Stats) error {
	return nil
}
