// +build darwin

package water

// PlatformSpecificParams defines parameters in Config that are specific to
// macOS. A zero-value of such type is valid, yielding an interface
// with OS defined name.
// Currently it is not possible to set the interface name in macOS.
type PlatformSpecificParams struct {
}

func defaultPlatformSpecificParams() PlatformSpecificParams {
	return PlatformSpecificParams{}
}
