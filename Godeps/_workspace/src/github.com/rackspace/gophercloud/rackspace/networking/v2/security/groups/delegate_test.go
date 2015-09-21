package groups

import (
	"fmt"
	"net/http"
	"testing"

	fake "github.com/rackspace/gophercloud/openstack/networking/v2/common"
	osGroups "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/security/groups"
	osRules "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/security/rules"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/security-groups", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
        {
          "security_groups": [
          {
            "description": "default",
            "id": "85cc3048-abc3-43cc-89b3-377341426ac5",
            "name": "default",
            "security_group_rules": [],
            "tenant_id": "e4f50856753b4dc6afee5fa6b9b6c550"
          }
          ]
        }
        `)
	})

	count := 0

	List(fake.ServiceClient(), osGroups.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := osGroups.ExtractGroups(page)
		if err != nil {
			t.Errorf("Failed to extract secgroups: %v", err)
			return false, err
		}

		expected := []osGroups.SecGroup{
			osGroups.SecGroup{
				Description: "default",
				ID:          "85cc3048-abc3-43cc-89b3-377341426ac5",
				Name:        "default",
				Rules:       []osRules.SecGroupRule{},
				TenantID:    "e4f50856753b4dc6afee5fa6b9b6c550",
			},
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/security-groups", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
      {
        "security_group": {
          "name": "new-webservers",
          "description": "security group for webservers"
        }
      }
    `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `
    {
      "security_group": {
        "description": "security group for webservers",
        "id": "2076db17-a522-4506-91de-c6dd8e837028",
        "name": "new-webservers",
        "security_group_rules": [
        {
          "direction": "egress",
          "ethertype": "IPv4",
          "id": "38ce2d8e-e8f1-48bd-83c2-d33cb9f50c3d",
          "port_range_max": null,
          "port_range_min": null,
          "protocol": null,
          "remote_group_id": null,
          "remote_ip_prefix": null,
          "security_group_id": "2076db17-a522-4506-91de-c6dd8e837028",
          "tenant_id": "e4f50856753b4dc6afee5fa6b9b6c550"
        },
        {
          "direction": "egress",
          "ethertype": "IPv6",
          "id": "565b9502-12de-4ffd-91e9-68885cff6ae1",
          "port_range_max": null,
          "port_range_min": null,
          "protocol": null,
          "remote_group_id": null,
          "remote_ip_prefix": null,
          "security_group_id": "2076db17-a522-4506-91de-c6dd8e837028",
          "tenant_id": "e4f50856753b4dc6afee5fa6b9b6c550"
        }
        ],
        "tenant_id": "e4f50856753b4dc6afee5fa6b9b6c550"
      }
    }
    `)
	})

	opts := osGroups.CreateOpts{Name: "new-webservers", Description: "security group for webservers"}
	_, err := Create(fake.ServiceClient(), opts).Extract()
	th.AssertNoErr(t, err)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/security-groups/85cc3048-abc3-43cc-89b3-377341426ac5", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
    {
      "security_group": {
        "description": "default",
        "id": "85cc3048-abc3-43cc-89b3-377341426ac5",
        "name": "default",
        "security_group_rules": [
        {
          "direction": "egress",
          "ethertype": "IPv6",
          "id": "3c0e45ff-adaf-4124-b083-bf390e5482ff",
          "port_range_max": null,
          "port_range_min": null,
          "protocol": null,
          "remote_group_id": null,
          "remote_ip_prefix": null,
          "security_group_id": "85cc3048-abc3-43cc-89b3-377341426ac5",
          "tenant_id": "e4f50856753b4dc6afee5fa6b9b6c550"
          },
        {
          "direction": "egress",
          "ethertype": "IPv4",
          "id": "93aa42e5-80db-4581-9391-3a608bd0e448",
          "port_range_max": null,
          "port_range_min": null,
          "protocol": null,
          "remote_group_id": null,
          "remote_ip_prefix": null,
          "security_group_id": "85cc3048-abc3-43cc-89b3-377341426ac5",
          "tenant_id": "e4f50856753b4dc6afee5fa6b9b6c550"
        }
        ],
        "tenant_id": "e4f50856753b4dc6afee5fa6b9b6c550"
      }
    }
  `)
	})

	sg, err := Get(fake.ServiceClient(), "85cc3048-abc3-43cc-89b3-377341426ac5").Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "default", sg.Description)
	th.AssertEquals(t, "85cc3048-abc3-43cc-89b3-377341426ac5", sg.ID)
	th.AssertEquals(t, "default", sg.Name)
	th.AssertEquals(t, 2, len(sg.Rules))
	th.AssertEquals(t, "e4f50856753b4dc6afee5fa6b9b6c550", sg.TenantID)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/security-groups/4ec89087-d057-4e2c-911f-60a3b47ee304", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})

	res := Delete(fake.ServiceClient(), "4ec89087-d057-4e2c-911f-60a3b47ee304")
	th.AssertNoErr(t, res.Err)
}
