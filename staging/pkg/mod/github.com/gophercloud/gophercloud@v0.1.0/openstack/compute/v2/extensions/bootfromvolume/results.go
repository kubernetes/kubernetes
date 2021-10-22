package bootfromvolume

import (
	os "github.com/gophercloud/gophercloud/openstack/compute/v2/servers"
)

// CreateResult temporarily contains the response from a Create call.
// It embeds the standard servers.CreateResults type and so can be used the
// same way as a standard server request result.
type CreateResult struct {
	os.CreateResult
}
