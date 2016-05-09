// +build linux

package libcontainer

import (
	"fmt"
	"os"
	"strings"

	"github.com/syndtr/gocapability/capability"
)

const allCapabilityTypes = capability.CAPS | capability.BOUNDS

var capabilityMap map[string]capability.Cap

func init() {
	capabilityMap = make(map[string]capability.Cap)
	last := capability.CAP_LAST_CAP
	// workaround for RHEL6 which has no /proc/sys/kernel/cap_last_cap
	if last == capability.Cap(63) {
		last = capability.CAP_BLOCK_SUSPEND
	}
	for _, cap := range capability.List() {
		if cap > last {
			continue
		}
		capKey := fmt.Sprintf("CAP_%s", strings.ToUpper(cap.String()))
		capabilityMap[capKey] = cap
	}
}

func newCapWhitelist(caps []string) (*whitelist, error) {
	l := []capability.Cap{}
	for _, c := range caps {
		v, ok := capabilityMap[c]
		if !ok {
			return nil, fmt.Errorf("unknown capability %q", c)
		}
		l = append(l, v)
	}
	pid, err := capability.NewPid(os.Getpid())
	if err != nil {
		return nil, err
	}
	return &whitelist{
		keep: l,
		pid:  pid,
	}, nil
}

type whitelist struct {
	pid  capability.Capabilities
	keep []capability.Cap
}

// dropBoundingSet drops the capability bounding set to those specified in the whitelist.
func (w *whitelist) dropBoundingSet() error {
	w.pid.Clear(capability.BOUNDS)
	w.pid.Set(capability.BOUNDS, w.keep...)
	return w.pid.Apply(capability.BOUNDS)
}

// drop drops all capabilities for the current process except those specified in the whitelist.
func (w *whitelist) drop() error {
	w.pid.Clear(allCapabilityTypes)
	w.pid.Set(allCapabilityTypes, w.keep...)
	return w.pid.Apply(allCapabilityTypes)
}
