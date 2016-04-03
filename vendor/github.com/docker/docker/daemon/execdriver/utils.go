package execdriver

import (
	"fmt"
	"strings"

	"github.com/docker/docker/pkg/stringutils"
	"github.com/syndtr/gocapability/capability"
)

var capabilityList Capabilities

func init() {
	last := capability.CAP_LAST_CAP
	// hack for RHEL6 which has no /proc/sys/kernel/cap_last_cap
	if last == capability.Cap(63) {
		last = capability.CAP_BLOCK_SUSPEND
	}
	for _, cap := range capability.List() {
		if cap > last {
			continue
		}
		capabilityList = append(capabilityList,
			&CapabilityMapping{
				Key:   strings.ToUpper(cap.String()),
				Value: cap,
			},
		)
	}
}

type (
	CapabilityMapping struct {
		Key   string         `json:"key,omitempty"`
		Value capability.Cap `json:"value,omitempty"`
	}
	Capabilities []*CapabilityMapping
)

func (c *CapabilityMapping) String() string {
	return c.Key
}

func GetCapability(key string) *CapabilityMapping {
	for _, capp := range capabilityList {
		if capp.Key == key {
			cpy := *capp
			return &cpy
		}
	}
	return nil
}

func GetAllCapabilities() []string {
	output := make([]string, len(capabilityList))
	for i, capability := range capabilityList {
		output[i] = capability.String()
	}
	return output
}

func TweakCapabilities(basics, adds, drops []string) ([]string, error) {
	var (
		newCaps []string
		allCaps = GetAllCapabilities()
	)

	// look for invalid cap in the drop list
	for _, cap := range drops {
		if strings.ToLower(cap) == "all" {
			continue
		}
		if !stringutils.InSlice(allCaps, cap) {
			return nil, fmt.Errorf("Unknown capability drop: %q", cap)
		}
	}

	// handle --cap-add=all
	if stringutils.InSlice(adds, "all") {
		basics = allCaps
	}

	if !stringutils.InSlice(drops, "all") {
		for _, cap := range basics {
			// skip `all` aready handled above
			if strings.ToLower(cap) == "all" {
				continue
			}

			// if we don't drop `all`, add back all the non-dropped caps
			if !stringutils.InSlice(drops, cap) {
				newCaps = append(newCaps, strings.ToUpper(cap))
			}
		}
	}

	for _, cap := range adds {
		// skip `all` aready handled above
		if strings.ToLower(cap) == "all" {
			continue
		}

		if !stringutils.InSlice(allCaps, cap) {
			return nil, fmt.Errorf("Unknown capability to add: %q", cap)
		}

		// add cap if not already in the list
		if !stringutils.InSlice(newCaps, cap) {
			newCaps = append(newCaps, strings.ToUpper(cap))
		}
	}

	return newCaps, nil
}
