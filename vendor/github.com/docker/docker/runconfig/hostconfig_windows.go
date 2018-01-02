package runconfig

import (
	"fmt"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/pkg/sysinfo"
)

// DefaultDaemonNetworkMode returns the default network stack the daemon should
// use.
func DefaultDaemonNetworkMode() container.NetworkMode {
	return container.NetworkMode("nat")
}

// IsPreDefinedNetwork indicates if a network is predefined by the daemon
func IsPreDefinedNetwork(network string) bool {
	return !container.NetworkMode(network).IsUserDefined()
}

// validateNetMode ensures that the various combinations of requested
// network settings are valid.
func validateNetMode(c *container.Config, hc *container.HostConfig) error {
	if hc == nil {
		return nil
	}

	err := validateNetContainerMode(c, hc)
	if err != nil {
		return err
	}

	if hc.NetworkMode.IsContainer() && hc.Isolation.IsHyperV() {
		return fmt.Errorf("Using the network stack of another container is not supported while using Hyper-V Containers")
	}

	return nil
}

// validateIsolation performs platform specific validation of the
// isolation in the hostconfig structure. Windows supports 'default' (or
// blank), 'process', or 'hyperv'.
func validateIsolation(hc *container.HostConfig) error {
	// We may not be passed a host config, such as in the case of docker commit
	if hc == nil {
		return nil
	}
	if !hc.Isolation.IsValid() {
		return fmt.Errorf("Invalid isolation: %q. Windows supports 'default', 'process', or 'hyperv'", hc.Isolation)
	}
	return nil
}

// validateQoS performs platform specific validation of the Qos settings
func validateQoS(hc *container.HostConfig) error {
	return nil
}

// validateResources performs platform specific validation of the resource settings
func validateResources(hc *container.HostConfig, si *sysinfo.SysInfo) error {
	// We may not be passed a host config, such as in the case of docker commit
	if hc == nil {
		return nil
	}
	if hc.Resources.CPURealtimePeriod != 0 {
		return fmt.Errorf("Windows does not support CPU real-time period")
	}
	if hc.Resources.CPURealtimeRuntime != 0 {
		return fmt.Errorf("Windows does not support CPU real-time runtime")
	}
	return nil
}

// validatePrivileged performs platform specific validation of the Privileged setting
func validatePrivileged(hc *container.HostConfig) error {
	// We may not be passed a host config, such as in the case of docker commit
	if hc == nil {
		return nil
	}
	if hc.Privileged {
		return fmt.Errorf("Windows does not support privileged mode")
	}
	return nil
}

// validateReadonlyRootfs performs platform specific validation of the ReadonlyRootfs setting
func validateReadonlyRootfs(hc *container.HostConfig) error {
	// We may not be passed a host config, such as in the case of docker commit
	if hc == nil {
		return nil
	}
	if hc.ReadonlyRootfs {
		return fmt.Errorf("Windows does not support root filesystem in read-only mode")
	}
	return nil
}
