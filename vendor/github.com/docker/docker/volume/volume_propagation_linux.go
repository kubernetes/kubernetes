// +build linux

package volume

import (
	"strings"

	mounttypes "github.com/docker/docker/api/types/mount"
)

// DefaultPropagationMode defines what propagation mode should be used by
// default if user has not specified one explicitly.
// propagation modes
const DefaultPropagationMode = mounttypes.PropagationRPrivate

var propagationModes = map[mounttypes.Propagation]bool{
	mounttypes.PropagationPrivate:  true,
	mounttypes.PropagationRPrivate: true,
	mounttypes.PropagationSlave:    true,
	mounttypes.PropagationRSlave:   true,
	mounttypes.PropagationShared:   true,
	mounttypes.PropagationRShared:  true,
}

// GetPropagation extracts and returns the mount propagation mode. If there
// are no specifications, then by default it is "private".
func GetPropagation(mode string) mounttypes.Propagation {
	for _, o := range strings.Split(mode, ",") {
		prop := mounttypes.Propagation(o)
		if propagationModes[prop] {
			return prop
		}
	}
	return DefaultPropagationMode
}

// HasPropagation checks if there is a valid propagation mode present in
// passed string. Returns true if a valid propagation mode specifier is
// present, false otherwise.
func HasPropagation(mode string) bool {
	for _, o := range strings.Split(mode, ",") {
		if propagationModes[mounttypes.Propagation(o)] {
			return true
		}
	}
	return false
}
