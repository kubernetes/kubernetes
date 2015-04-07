package capabilities

import (
	"os"

	"github.com/syndtr/gocapability/capability"
)

const allCapabilityTypes = capability.CAPS | capability.BOUNDS

// DropBoundingSet drops the capability bounding set to those specified in the
// container configuration.
func DropBoundingSet(capabilities []string) error {
	c, err := capability.NewPid(os.Getpid())
	if err != nil {
		return err
	}

	keep := getEnabledCapabilities(capabilities)
	c.Clear(capability.BOUNDS)
	c.Set(capability.BOUNDS, keep...)

	if err := c.Apply(capability.BOUNDS); err != nil {
		return err
	}

	return nil
}

// DropCapabilities drops all capabilities for the current process except those specified in the container configuration.
func DropCapabilities(capList []string) error {
	c, err := capability.NewPid(os.Getpid())
	if err != nil {
		return err
	}

	keep := getEnabledCapabilities(capList)
	c.Clear(allCapabilityTypes)
	c.Set(allCapabilityTypes, keep...)

	if err := c.Apply(allCapabilityTypes); err != nil {
		return err
	}
	return nil
}

// getEnabledCapabilities returns the capabilities that should not be dropped by the container.
func getEnabledCapabilities(capList []string) []capability.Cap {
	keep := []capability.Cap{}
	for _, capability := range capList {
		if c := GetCapability(capability); c != nil {
			keep = append(keep, c.Value)
		}
	}
	return keep
}
