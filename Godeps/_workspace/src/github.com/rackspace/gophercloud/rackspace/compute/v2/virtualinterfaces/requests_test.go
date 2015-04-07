package virtualinterfaces

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/servers/12345/os-virtual-interfacesv2", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "virtual_interfaces": [
        {
            "id": "de7c6d53-b895-4b4a-963c-517ccb0f0775",
            "ip_addresses": [
                {
                    "address": "192.168.0.2",
                    "network_id": "f212726e-6321-4210-9bae-a13f5a33f83f",
                    "network_label": "superprivate_xml"
                }
            ],
            "mac_address": "BC:76:4E:04:85:20"
        },
        {
            "id": "e14e789d-3b98-44a6-9c2d-c23eb1d1465c",
            "ip_addresses": [
                {
                    "address": "10.181.1.30",
                    "network_id": "3b324a1b-31b8-4db5-9fe5-4a2067f60297",
                    "network_label": "private"
                }
            ],
            "mac_address": "BC:76:4E:04:81:55"
        }
    ]
}
      `)
	})

	client := fake.ServiceClient()
	count := 0

	err := List(client, "12345").EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractVirtualInterfaces(page)
		if err != nil {
			t.Errorf("Failed to extract networks: %v", err)
			return false, err
		}

		expected := []VirtualInterface{
			VirtualInterface{
				MACAddress: "BC:76:4E:04:85:20",
				IPAddresses: []IPAddress{
					IPAddress{
						Address:      "192.168.0.2",
						NetworkID:    "f212726e-6321-4210-9bae-a13f5a33f83f",
						NetworkLabel: "superprivate_xml",
					},
				},
				ID: "de7c6d53-b895-4b4a-963c-517ccb0f0775",
			},
			VirtualInterface{
				MACAddress: "BC:76:4E:04:81:55",
				IPAddresses: []IPAddress{
					IPAddress{
						Address:      "10.181.1.30",
						NetworkID:    "3b324a1b-31b8-4db5-9fe5-4a2067f60297",
						NetworkLabel: "private",
					},
				},
				ID: "e14e789d-3b98-44a6-9c2d-c23eb1d1465c",
			},
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, 1, count)
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/servers/12345/os-virtual-interfacesv2", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "virtual_interface": {
        "network_id": "6789"
    }
}
      `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `{
      "virtual_interfaces": [
        {
          "id": "de7c6d53-b895-4b4a-963c-517ccb0f0775",
          "ip_addresses": [
            {
              "address": "192.168.0.2",
              "network_id": "f212726e-6321-4210-9bae-a13f5a33f83f",
              "network_label": "superprivate_xml"
            }
          ],
          "mac_address": "BC:76:4E:04:85:20"
        }
      ]
    }`)
	})

	expected := &VirtualInterface{
		MACAddress: "BC:76:4E:04:85:20",
		IPAddresses: []IPAddress{
			IPAddress{
				Address:      "192.168.0.2",
				NetworkID:    "f212726e-6321-4210-9bae-a13f5a33f83f",
				NetworkLabel: "superprivate_xml",
			},
		},
		ID: "de7c6d53-b895-4b4a-963c-517ccb0f0775",
	}

	actual, err := Create(fake.ServiceClient(), "12345", "6789").Extract()
	th.AssertNoErr(t, err)

	th.CheckDeepEquals(t, expected, actual)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/servers/12345/os-virtual-interfacesv2/6789", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})

	res := Delete(fake.ServiceClient(), "12345", "6789")
	th.AssertNoErr(t, res.Err)
}
