// +build fixtures

package quotasets

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

// GetOutput is a sample response to a Get call.
const GetOutput = `
{
   "quota_set" : {
      "instances" : 25,
      "security_groups" : 10,
      "security_group_rules" : 20,
      "cores" : 200,
      "injected_file_content_bytes" : 10240,
      "injected_files" : 5,
      "metadata_items" : 128,
      "ram" : 200000,
      "keypairs" : 10,
      "injected_file_path_bytes" : 255
   }
}
`

const FirstTenantID = "555544443333222211110000ffffeeee"

// FirstQuotaSet is the first result in ListOutput.
var FirstQuotaSet = QuotaSet{
	FixedIps:                 0,
	FloatingIps:              0,
	InjectedFileContentBytes: 10240,
	InjectedFilePathBytes:    255,
	InjectedFiles:            5,
	KeyPairs:                 10,
	MetadataItems:            128,
	Ram:                      200000,
	SecurityGroupRules:       20,
	SecurityGroups:           10,
	Cores:                    200,
	Instances:                25,
}

// HandleGetSuccessfully configures the test server to respond to a Get request for sample tenant
func HandleGetSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/os-quota-sets/"+FirstTenantID, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, GetOutput)
	})
}
