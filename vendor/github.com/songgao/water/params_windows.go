// +build windows

package water

// PlatformSpecificParams defines parameters in Config that are specific to
// Windows. A zero-value of such type is valid.
type PlatformSpecificParams struct {
	// ComponentID associates with the virtual adapter that exists in Windows.
	// This is usually configured when driver for the adapter is installed. A
	// zero-value of this field, i.e., an empty string, causes the interface to
	// use the default ComponentId. The default ComponentId is set to tap0901,
	// the one used by OpenVPN.
	ComponentID string
	// Network is required when creating a TUN interface. The library will call
	// net.ParseCIDR() to parse this string into LocalIP, RemoteNetaddr,
	// RemoteNetmask. The underlying driver will need those to generate ARP
	// response to Windows kernel, to emulate an TUN interface.
	// Please note that it cannot perceive the IP changes caused by DHCP, user
	// configuration to the adapter and etc,. If IP changed, please reconfigure
	// the adapter using syscall, just like openDev().
	// For detail, please refer
	// https://github.com/OpenVPN/tap-windows6/blob/master/src/device.c#L431
	// and https://github.com/songgao/water/pull/13#issuecomment-270341777
	Network string
}

func defaultPlatformSpecificParams() PlatformSpecificParams {
	return PlatformSpecificParams{
		ComponentID: "tap0901",
		Network:     "192.168.1.10/24",
	}
}
