// +build linux

package capabilities

import (
	"sort"
	"strings"

	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/sirupsen/logrus"
	"github.com/syndtr/gocapability/capability"
)

const allCapabilityTypes = capability.CAPS | capability.BOUNDING | capability.AMBIENT

var (
	capabilityMap map[string]capability.Cap
	capTypes      = []capability.CapType{
		capability.BOUNDING,
		capability.PERMITTED,
		capability.INHERITABLE,
		capability.EFFECTIVE,
		capability.AMBIENT,
	}
)

func init() {
	capabilityMap = make(map[string]capability.Cap, capability.CAP_LAST_CAP+1)
	for _, c := range capability.List() {
		if c > capability.CAP_LAST_CAP {
			continue
		}
		capabilityMap["CAP_"+strings.ToUpper(c.String())] = c
	}
}

// New creates a new Caps from the given Capabilities config. Unknown Capabilities
// or Capabilities that are unavailable in the current environment are ignored,
// printing a warning instead.
func New(capConfig *configs.Capabilities) (*Caps, error) {
	var (
		err error
		c   Caps
	)

	unknownCaps := make(map[string]struct{})
	c.caps = map[capability.CapType][]capability.Cap{
		capability.BOUNDING:    capSlice(capConfig.Bounding, unknownCaps),
		capability.EFFECTIVE:   capSlice(capConfig.Effective, unknownCaps),
		capability.INHERITABLE: capSlice(capConfig.Inheritable, unknownCaps),
		capability.PERMITTED:   capSlice(capConfig.Permitted, unknownCaps),
		capability.AMBIENT:     capSlice(capConfig.Ambient, unknownCaps),
	}
	if c.pid, err = capability.NewPid2(0); err != nil {
		return nil, err
	}
	if err = c.pid.Load(); err != nil {
		return nil, err
	}
	if len(unknownCaps) > 0 {
		logrus.Warn("ignoring unknown or unavailable capabilities: ", mapKeys(unknownCaps))
	}
	return &c, nil
}

// capSlice converts the slice of capability names in caps, to their numeric
// equivalent, and returns them as a slice. Unknown or unavailable capabilities
// are not returned, but appended to unknownCaps.
func capSlice(caps []string, unknownCaps map[string]struct{}) []capability.Cap {
	var out []capability.Cap
	for _, c := range caps {
		if v, ok := capabilityMap[c]; !ok {
			unknownCaps[c] = struct{}{}
		} else {
			out = append(out, v)
		}
	}
	return out
}

// mapKeys returns the keys of input in sorted order
func mapKeys(input map[string]struct{}) []string {
	var keys []string
	for c := range input {
		keys = append(keys, c)
	}
	sort.Strings(keys)
	return keys
}

// Caps holds the capabilities for a container.
type Caps struct {
	pid  capability.Capabilities
	caps map[capability.CapType][]capability.Cap
}

// ApplyBoundingSet sets the capability bounding set to those specified in the whitelist.
func (c *Caps) ApplyBoundingSet() error {
	c.pid.Clear(capability.BOUNDING)
	c.pid.Set(capability.BOUNDING, c.caps[capability.BOUNDING]...)
	return c.pid.Apply(capability.BOUNDING)
}

// Apply sets all the capabilities for the current process in the config.
func (c *Caps) ApplyCaps() error {
	c.pid.Clear(allCapabilityTypes)
	for _, g := range capTypes {
		c.pid.Set(g, c.caps[g]...)
	}
	return c.pid.Apply(allCapabilityTypes)
}
