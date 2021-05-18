// +build linux

package capabilities

import (
	"fmt"
	"strings"

	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/syndtr/gocapability/capability"
)

const allCapabilityTypes = capability.CAPS | capability.BOUNDS | capability.AMBS

var capabilityMap map[string]capability.Cap

func init() {
	capabilityMap = make(map[string]capability.Cap, capability.CAP_LAST_CAP+1)
	for _, c := range capability.List() {
		if c > capability.CAP_LAST_CAP {
			continue
		}
		capabilityMap["CAP_"+strings.ToUpper(c.String())] = c
	}
}

// New creates a new Caps from the given Capabilities config.
func New(capConfig *configs.Capabilities) (*Caps, error) {
	var (
		err  error
		caps Caps
	)

	if caps.bounding, err = capSlice(capConfig.Bounding); err != nil {
		return nil, err
	}
	if caps.effective, err = capSlice(capConfig.Effective); err != nil {
		return nil, err
	}
	if caps.inheritable, err = capSlice(capConfig.Inheritable); err != nil {
		return nil, err
	}
	if caps.permitted, err = capSlice(capConfig.Permitted); err != nil {
		return nil, err
	}
	if caps.ambient, err = capSlice(capConfig.Ambient); err != nil {
		return nil, err
	}
	if caps.pid, err = capability.NewPid2(0); err != nil {
		return nil, err
	}
	if err = caps.pid.Load(); err != nil {
		return nil, err
	}
	return &caps, nil
}

func capSlice(caps []string) ([]capability.Cap, error) {
	out := make([]capability.Cap, len(caps))
	for i, c := range caps {
		v, ok := capabilityMap[c]
		if !ok {
			return nil, fmt.Errorf("unknown capability %q", c)
		}
		out[i] = v
	}
	return out, nil
}

// Caps holds the capabilities for a container.
type Caps struct {
	pid         capability.Capabilities
	bounding    []capability.Cap
	effective   []capability.Cap
	inheritable []capability.Cap
	permitted   []capability.Cap
	ambient     []capability.Cap
}

// ApplyBoundingSet sets the capability bounding set to those specified in the whitelist.
func (c *Caps) ApplyBoundingSet() error {
	c.pid.Clear(capability.BOUNDS)
	c.pid.Set(capability.BOUNDS, c.bounding...)
	return c.pid.Apply(capability.BOUNDS)
}

// Apply sets all the capabilities for the current process in the config.
func (c *Caps) ApplyCaps() error {
	c.pid.Clear(allCapabilityTypes)
	c.pid.Set(capability.BOUNDS, c.bounding...)
	c.pid.Set(capability.PERMITTED, c.permitted...)
	c.pid.Set(capability.INHERITABLE, c.inheritable...)
	c.pid.Set(capability.EFFECTIVE, c.effective...)
	c.pid.Set(capability.AMBIENT, c.ambient...)
	return c.pid.Apply(allCapabilityTypes)
}
