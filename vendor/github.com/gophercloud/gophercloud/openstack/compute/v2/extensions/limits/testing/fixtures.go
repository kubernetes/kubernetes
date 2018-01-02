package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/limits"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// GetOutput is a sample response to a Get call.
const GetOutput = `
{
    "limits": {
        "rate": [],
        "absolute": {
            "maxServerMeta": 128,
            "maxPersonality": 5,
            "totalServerGroupsUsed": 0,
            "maxImageMeta": 128,
            "maxPersonalitySize": 10240,
            "maxTotalKeypairs": 100,
            "maxSecurityGroupRules": 20,
            "maxServerGroups": 10,
            "totalCoresUsed": 1,
            "totalRAMUsed": 2048,
            "totalInstancesUsed": 1,
            "maxSecurityGroups": 10,
            "totalFloatingIpsUsed": 0,
            "maxTotalCores": 20,
            "maxServerGroupMembers": 10,
            "maxTotalFloatingIps": 10,
            "totalSecurityGroupsUsed": 1,
            "maxTotalInstances": 10,
            "maxTotalRAMSize": 51200
        }
    }
}
`

// LimitsResult is the result of the limits in GetOutput.
var LimitsResult = limits.Limits{
	Absolute: limits.Absolute{
		MaxServerMeta:           128,
		MaxPersonality:          5,
		TotalServerGroupsUsed:   0,
		MaxImageMeta:            128,
		MaxPersonalitySize:      10240,
		MaxTotalKeypairs:        100,
		MaxSecurityGroupRules:   20,
		MaxServerGroups:         10,
		TotalCoresUsed:          1,
		TotalRAMUsed:            2048,
		TotalInstancesUsed:      1,
		MaxSecurityGroups:       10,
		TotalFloatingIpsUsed:    0,
		MaxTotalCores:           20,
		MaxServerGroupMembers:   10,
		MaxTotalFloatingIps:     10,
		TotalSecurityGroupsUsed: 1,
		MaxTotalInstances:       10,
		MaxTotalRAMSize:         51200,
	},
}

const TenantID = "555544443333222211110000ffffeeee"

// HandleGetSuccessfully configures the test server to respond to a Get request
// for a limit.
func HandleGetSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/limits", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, GetOutput)
	})
}
