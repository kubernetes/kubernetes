package system

import "os"

// LCOWSupported determines if Linux Containers on Windows are supported.
// Note: This feature is in development (06/17) and enabled through an
// environment variable. At a future time, it will be enabled based
// on build number. @jhowardmsft
var lcowSupported = false

func init() {
	// LCOW initialization
	if os.Getenv("LCOW_SUPPORTED") != "" {
		lcowSupported = true
	}

}
