package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/attachinterfaces"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

// ListInterfacesExpected represents an expected repsonse from a ListInterfaces request.
var ListInterfacesExpected = []attachinterfaces.Interface{
	{
		PortState: "ACTIVE",
		FixedIPs: []attachinterfaces.FixedIP{
			{
				SubnetID:  "d7906db4-a566-4546-b1f4-5c7fa70f0bf3",
				IPAddress: "10.0.0.7",
			},
			{
				SubnetID:  "45906d64-a548-4276-h1f8-kcffa80fjbnl",
				IPAddress: "10.0.0.8",
			},
		},
		PortID:  "0dde1598-b374-474e-986f-5b8dd1df1d4e",
		NetID:   "8a5fe506-7e9f-4091-899b-96336909d93c",
		MACAddr: "fa:16:3e:38:2d:80",
	},
}

// HandleInterfaceListSuccessfully sets up the test server to respond to a ListInterfaces request.
func HandleInterfaceListSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/servers/b07e7a3b-d951-4efc-a4f9-ac9f001afb7f/os-interface", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", client.TokenID)

		w.Header().Add("Content-Type", "application/json")
		fmt.Fprintf(w, `{
			"interfaceAttachments": [
				{
					"port_state":"ACTIVE",
					"fixed_ips": [
						{
							"subnet_id": "d7906db4-a566-4546-b1f4-5c7fa70f0bf3",
							"ip_address": "10.0.0.7"
						},
						{
							"subnet_id": "45906d64-a548-4276-h1f8-kcffa80fjbnl",
							"ip_address": "10.0.0.8"
						}
					],
					"port_id": "0dde1598-b374-474e-986f-5b8dd1df1d4e",
					"net_id": "8a5fe506-7e9f-4091-899b-96336909d93c",
					"mac_addr": "fa:16:3e:38:2d:80"
				}
			]
		}`)
	})
}
