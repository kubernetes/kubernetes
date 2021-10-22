package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud"
	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/fwaas/firewalls"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/fwaas/routerinsertion"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/fw/firewalls", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "firewall":{
        "name": "fw",
        "description": "OpenStack firewall",
        "admin_state_up": true,
        "firewall_policy_id": "19ab8c87-4a32-4e6a-a74e-b77fffb89a0c",
        "tenant_id": "b4eedccc6fb74fa8a7ad6b08382b852b",
        "router_ids": [
          "8a3a0d6a-34b5-4a92-b65d-6375a4c1e9e8"
        ]
    }
}
      `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `
{
    "firewall":{
        "status": "PENDING_CREATE",
        "name": "fw",
        "description": "OpenStack firewall",
        "admin_state_up": true,
        "tenant_id": "b4eedccc6fb74fa8a7ad6b08382b852b",
        "firewall_policy_id": "19ab8c87-4a32-4e6a-a74e-b77fffb89a0c"
    }
}
    `)
	})

	firewallCreateOpts := firewalls.CreateOpts{
		TenantID:     "b4eedccc6fb74fa8a7ad6b08382b852b",
		Name:         "fw",
		Description:  "OpenStack firewall",
		AdminStateUp: gophercloud.Enabled,
		PolicyID:     "19ab8c87-4a32-4e6a-a74e-b77fffb89a0c",
	}
	createOpts := routerinsertion.CreateOptsExt{
		CreateOptsBuilder: firewallCreateOpts,
		RouterIDs:         []string{"8a3a0d6a-34b5-4a92-b65d-6375a4c1e9e8"},
	}

	_, err := firewalls.Create(fake.ServiceClient(), createOpts).Extract()
	th.AssertNoErr(t, err)
}

func TestCreateWithNoRouters(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/fw/firewalls", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "firewall":{
        "name": "fw",
        "description": "OpenStack firewall",
        "admin_state_up": true,
        "firewall_policy_id": "19ab8c87-4a32-4e6a-a74e-b77fffb89a0c",
        "tenant_id": "b4eedccc6fb74fa8a7ad6b08382b852b",
        "router_ids": []
    }
}
      `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `
{
    "firewall":{
        "status": "PENDING_CREATE",
        "name": "fw",
        "description": "OpenStack firewall",
        "admin_state_up": true,
        "tenant_id": "b4eedccc6fb74fa8a7ad6b08382b852b",
        "firewall_policy_id": "19ab8c87-4a32-4e6a-a74e-b77fffb89a0c"
    }
}
    `)
	})

	firewallCreateOpts := firewalls.CreateOpts{
		TenantID:     "b4eedccc6fb74fa8a7ad6b08382b852b",
		Name:         "fw",
		Description:  "OpenStack firewall",
		AdminStateUp: gophercloud.Enabled,
		PolicyID:     "19ab8c87-4a32-4e6a-a74e-b77fffb89a0c",
	}
	createOpts := routerinsertion.CreateOptsExt{
		CreateOptsBuilder: firewallCreateOpts,
		RouterIDs:         []string{},
	}

	_, err := firewalls.Create(fake.ServiceClient(), createOpts).Extract()
	th.AssertNoErr(t, err)
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/fw/firewalls/ea5b5315-64f6-4ea3-8e58-981cc37c6576", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "firewall":{
        "name": "fw",
        "description": "updated fw",
        "admin_state_up":false,
        "firewall_policy_id": "19ab8c87-4a32-4e6a-a74e-b77fffb89a0c",
        "router_ids": [
          "8a3a0d6a-34b5-4a92-b65d-6375a4c1e9e8"
        ]
    }
}
      `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "firewall": {
        "status": "ACTIVE",
        "name": "fw",
        "admin_state_up": false,
        "tenant_id": "b4eedccc6fb74fa8a7ad6b08382b852b",
        "firewall_policy_id": "19ab8c87-4a32-4e6a-a74e-b77fffb89a0c",
        "id": "ea5b5315-64f6-4ea3-8e58-981cc37c6576",
        "description": "OpenStack firewall"
    }
}
    `)
	})

	var name = "fw"
	var description = "updated fw"
	firewallUpdateOpts := firewalls.UpdateOpts{
		Name:         &name,
		Description:  &description,
		AdminStateUp: gophercloud.Disabled,
		PolicyID:     "19ab8c87-4a32-4e6a-a74e-b77fffb89a0c",
	}
	updateOpts := routerinsertion.UpdateOptsExt{
		UpdateOptsBuilder: firewallUpdateOpts,
		RouterIDs:         []string{"8a3a0d6a-34b5-4a92-b65d-6375a4c1e9e8"},
	}

	_, err := firewalls.Update(fake.ServiceClient(), "ea5b5315-64f6-4ea3-8e58-981cc37c6576", updateOpts).Extract()
	th.AssertNoErr(t, err)
}

func TestUpdateWithNoRouters(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/fw/firewalls/ea5b5315-64f6-4ea3-8e58-981cc37c6576", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "firewall":{
        "name": "fw",
        "description": "updated fw",
        "admin_state_up":false,
        "firewall_policy_id": "19ab8c87-4a32-4e6a-a74e-b77fffb89a0c",
        "router_ids": []
    }
}
      `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "firewall": {
        "status": "ACTIVE",
        "name": "fw",
        "admin_state_up": false,
        "tenant_id": "b4eedccc6fb74fa8a7ad6b08382b852b",
        "firewall_policy_id": "19ab8c87-4a32-4e6a-a74e-b77fffb89a0c",
        "id": "ea5b5315-64f6-4ea3-8e58-981cc37c6576",
        "description": "OpenStack firewall"
    }
}
    `)
	})

	var name = "fw"
	var description = "updated fw"
	firewallUpdateOpts := firewalls.UpdateOpts{
		Name:         &name,
		Description:  &description,
		AdminStateUp: gophercloud.Disabled,
		PolicyID:     "19ab8c87-4a32-4e6a-a74e-b77fffb89a0c",
	}
	updateOpts := routerinsertion.UpdateOptsExt{
		UpdateOptsBuilder: firewallUpdateOpts,
		RouterIDs:         []string{},
	}

	_, err := firewalls.Update(fake.ServiceClient(), "ea5b5315-64f6-4ea3-8e58-981cc37c6576", updateOpts).Extract()
	th.AssertNoErr(t, err)
}
