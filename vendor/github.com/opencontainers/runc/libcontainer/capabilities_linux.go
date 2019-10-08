// +build linux

package libcontainer

import (
	"fmt"
	"strings"

	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/syndtr/gocapability/capability"
)

const allCapabilityTypes = capability.CAPS | capability.BOUNDS | capability.AMBS

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

func newContainerCapList(capConfig *configs.Capabilities) (*containerCapabilities, error) {
	bounding := []capability.Cap{}
	for _, c := range capConfig.Bounding {
		v, ok := capabilityMap[c]
		if !ok {
			return nil, fmt.Errorf("unknown capability %q", c)
		}
		bounding = append(bounding, v)
	}
	effective := []capability.Cap{}
	for _, c := range capConfig.Effective {
		v, ok := capabilityMap[c]
		if !ok {
			return nil, fmt.Errorf("unknown capability %q", c)
		}
		effective = append(effective, v)
	}
	inheritable := []capability.Cap{}
	for _, c := range capConfig.Inheritable {
		v, ok := capabilityMap[c]
		if !ok {
			return nil, fmt.Errorf("unknown capability %q", c)
		}
		inheritable = append(inheritable, v)
	}
	permitted := []capability.Cap{}
	for _, c := range capConfig.Permitted {
		v, ok := capabilityMap[c]
		if !ok {
			return nil, fmt.Errorf("unknown capability %q", c)
		}
		permitted = append(permitted, v)
	}
	ambient := []capability.Cap{}
	for _, c := range capConfig.Ambient {
		v, ok := capabilityMap[c]
		if !ok {
			return nil, fmt.Errorf("unknown capability %q", c)
		}
		ambient = append(ambient, v)
	}
	pid, err := capability.NewPid2(0)
	if err != nil {
		return nil, err
	}
	err = pid.Load()
	if err != nil {
		return nil, err
	}
	return &containerCapabilities{
		bounding:    bounding,
		effective:   effective,
		inheritable: inheritable,
		permitted:   permitted,
		ambient:     ambient,
		pid:         pid,
	}, nil
}

type containerCapabilities struct {
	pid         capability.Capabilities
	bounding    []capability.Cap
	effective   []capability.Cap
	inheritable []capability.Cap
	permitted   []capability.Cap
	ambient     []capability.Cap
}

// ApplyBoundingSet sets the capability bounding set to those specified in the whitelist.
func (c *containerCapabilities) ApplyBoundingSet() error {
	c.pid.Clear(capability.BOUNDS)
	c.pid.Set(capability.BOUNDS, c.bounding...)
	return c.pid.Apply(capability.BOUNDS)
}

// Apply sets all the capabilities for the current process in the config.
func (c *containerCapabilities) ApplyCaps() error {
	c.pid.Clear(allCapabilityTypes)
	c.pid.Set(capability.BOUNDS, c.bounding...)
	c.pid.Set(capability.PERMITTED, c.permitted...)
	c.pid.Set(capability.INHERITABLE, c.inheritable...)
	c.pid.Set(capability.EFFECTIVE, c.effective...)
	c.pid.Set(capability.AMBIENT, c.ambient...)
	return c.pid.Apply(allCapabilityTypes)
}
