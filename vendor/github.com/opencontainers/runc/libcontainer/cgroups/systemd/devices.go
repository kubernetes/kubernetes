package systemd

import (
	"reflect"

	dbus "github.com/godbus/dbus/v5"

	"github.com/opencontainers/runc/libcontainer/configs"
)

// freezeBeforeSet answers whether there is a need to freeze the cgroup before
// applying its systemd unit properties, and thaw after, while avoiding
// unnecessary freezer state changes.
//
// The reason why we have to freeze is that systemd's application of device
// rules is done disruptively, resulting in spurious errors to common devices
// (unlike our fs driver, they will happily write deny-all rules to running
// containers). So we have to freeze the container to avoid the container get
// an occasional "permission denied" error.
func (m *LegacyManager) freezeBeforeSet(unitName string, r *configs.Resources) (needsFreeze, needsThaw bool, err error) {
	// Special case for SkipDevices, as used by Kubernetes to create pod
	// cgroups with allow-all device policy).
	if r.SkipDevices {
		if r.SkipFreezeOnSet {
			// Both needsFreeze and needsThaw are false.
			return
		}

		// No need to freeze if SkipDevices is set, and either
		// (1) systemd unit does not (yet) exist, or
		// (2) it has DevicePolicy=auto and empty DeviceAllow list.
		//
		// Interestingly, (1) and (2) are the same here because
		// a non-existent unit returns default properties,
		// and settings in (2) are the defaults.
		//
		// Do not return errors from getUnitTypeProperty, as they alone
		// should not prevent Set from working.

		unitType := getUnitType(unitName)

		devPolicy, e := getUnitTypeProperty(m.dbus, unitName, unitType, "DevicePolicy")
		if e == nil && devPolicy.Value == dbus.MakeVariant("auto") {
			devAllow, e := getUnitTypeProperty(m.dbus, unitName, unitType, "DeviceAllow")
			if e == nil {
				if rv := reflect.ValueOf(devAllow.Value.Value()); rv.Kind() == reflect.Slice && rv.Len() == 0 {
					needsFreeze = false
					needsThaw = false
					return
				}
			}
		}
	}

	needsFreeze = true
	needsThaw = true

	// Check the current freezer state.
	freezerState, err := m.GetFreezerState()
	if err != nil {
		return
	}
	if freezerState == configs.Frozen {
		// Already frozen, and should stay frozen.
		needsFreeze = false
		needsThaw = false
	}

	if r.Freezer == configs.Frozen {
		// Will be frozen anyway -- no need to thaw.
		needsThaw = false
	}
	return
}
