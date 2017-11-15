package system

import "os"

// lcowSupported determines if Linux Containers on Windows are supported.
var lcowSupported = false

// InitLCOW sets whether LCOW is supported or not
// TODO @jhowardmsft.
// 1. Replace with RS3 RTM build number.
// 2. Remove the getenv check when image-store is coalesced as shouldn't be needed anymore.
func InitLCOW(experimental bool) {
	v := GetOSVersion()
	if experimental && v.Build > 16270 && os.Getenv("LCOW_SUPPORTED") != "" {
		lcowSupported = true
	}
}
