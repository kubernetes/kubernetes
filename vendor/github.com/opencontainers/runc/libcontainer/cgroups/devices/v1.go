package devices

import (
	"bytes"
	"errors"
	"reflect"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runc/libcontainer/devices"
	"github.com/opencontainers/runc/libcontainer/userns"
)

var testingSkipFinalCheck bool

func setV1(path string, r *configs.Resources) error {
	if userns.RunningInUserNS() || r.SkipDevices {
		return nil
	}
	// Generate two emulators, one for the current state of the cgroup and one
	// for the requested state by the user.
	current, err := loadEmulator(path)
	if err != nil {
		return err
	}
	target, err := buildEmulator(r.Devices)
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
		if err := cgroups.WriteFile(path, file, rule.CgroupString()); err != nil {
			return err
		}
	}

	// Final safety check -- ensure that the resulting state is what was
	// requested. This is only really correct for white-lists, but for
	// black-lists we can at least check that the cgroup is in the right mode.
	//
	// This safety-check is skipped for the unit tests because we cannot
	// currently mock devices.list correctly.
	if !testingSkipFinalCheck {
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

func loadEmulator(path string) (*emulator, error) {
	list, err := cgroups.ReadFile(path, "devices.list")
	if err != nil {
		return nil, err
	}
	return emulatorFromList(bytes.NewBufferString(list))
}

func buildEmulator(rules []*devices.Rule) (*emulator, error) {
	// This defaults to a white-list -- which is what we want!
	emu := &emulator{}
	for _, rule := range rules {
		if err := emu.Apply(*rule); err != nil {
			return nil, err
		}
	}
	return emu, nil
}
