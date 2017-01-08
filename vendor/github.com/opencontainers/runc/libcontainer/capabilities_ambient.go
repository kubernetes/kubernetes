// +build linux,ambient

package libcontainer

import "github.com/syndtr/gocapability/capability"

const allCapabilityTypes = capability.CAPS | capability.BOUNDS | capability.AMBS
