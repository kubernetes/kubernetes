package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud"
	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/fwaas/firewalls"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/fwaas/routerinsertion"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/fw/firewalls", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
   "firewalls":[
        {
           "status": "ACTIVE",
           "name": "fw1",
           "admin_state_up": false,
           "tenant_id": "b4eedccc6fb74fa8a7ad6b08382b852b",
           "firewall_policy_id": "34be8c83-4d42-4dca-a74e-b77fffb8e28a",
           "id": "fb5b5315-64f6-4ea3-8e58-981cc37c6f61",
           "description": "OpenStack firewall 1"
        },
        {
           "status": "PENDING_UPDATE",
           "name": "fw2",
           "admin_state_up": true,
           "tenant_id": "b4eedccc6fb74fa8a7ad6b08382b852b",
           "firewall_policy_id": "34be8c83-4d42-4dca-a74e-b77fffb8e299",
           "id": "fb5b5315-64f6-4ea3-8e58-981cc37c6f99",
           "description": "OpenStack firewall 2"
        }
   ]
}
      `)
	})

	count := 0

	firewalls.List(fake.ServiceClient(), firewalls.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := firewalls.ExtractFirewalls(page)
		if err != nil {
			t.Errorf("Failed to extract members: %v", err)
			return false, err
		}

		expected := []firewalls.Firewall{
			{
				Status:       "ACTIVE",
				Name:         "fw1",
				AdminStateUp: false,
				TenantID:     "b4eedccc6fb74fa8a7ad6b08382b852b",
				PolicyID:     "34be8c83-4d42-4dca-a74e-b77fffb8e28a",
				ID:           "fb5b5315-64f6-4ea3-8e58-981cc37c6f61",
				Description:  "OpenStack firewall 1",
			},
			{
				Status:       "PENDING_UPDATE",
				Name:         "fw2",
				AdminStateUp: true,
				TenantID:     "b4eedccc6fb74fa8a7ad6b08382b852b",
				PolicyID:     "34be8c83-4d42-4dca-a74e-b77fffb8e299",
				ID:           "fb5b5315-64f6-4ea3-8e58-981cc37c6f99",
				Description:  "OpenStack firewall 2",
			},
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestListWithExtensions(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/fw/firewalls", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
   "firewalls":[
        {
           "status": "ACTIVE",
           "name": "fw1",
           "admin_state_up": false,
           "tenant_id": "b4eedccc6fb74fa8a7ad6b08382b852b",
           "firewall_policy_id": "34be8c83-4d42-4dca-a74e-b77fffb8e28a",
           "id": "fb5b5315-64f6-4ea3-8e58-981cc37c6f61",
           "description": "OpenStack firewall 1",
           "router_ids": ["abcd1234"]
        },
        {
           "status": "PENDING_UPDATE",
           "name": "fw2",
           "admin_state_up": true,
           "tenant_id": "b4eedccc6fb74fa8a7ad6b08382b852b",
           "firewall_policy_id": "34be8c83-4d42-4dca-a74e-b77fffb8e299",
           "id": "fb5b5315-64f6-4ea3-8e58-981cc37c6f99",
           "description": "OpenStack firewall 2"
        }
   ]
}
      `)
	})

	type FirewallsWithExt struct {
		firewalls.Firewall
		routerinsertion.FirewallExt
	}

	allPages, err := firewalls.List(fake.ServiceClient(), nil).AllPages()
	th.AssertNoErr(t, err)

	var actual []FirewallsWithExt
	err = firewalls.ExtractFirewallsInto(allPages, &actual)
	th.AssertNoErr(t, err)
	th.AssertEquals(t, 2, len(actual))
	th.AssertEquals(t, "fb5b5315-64f6-4ea3-8e58-981cc37c6f61", actual[0].ID)
	th.AssertEquals(t, "abcd1234", actual[0].RouterIDs[0])
}

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
        "tenant_id": "b4eedccc6fb74fa8a7ad6b08382b852b"
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

	options := firewalls.CreateOpts{
		TenantID:     "b4eedccc6fb74fa8a7ad6b08382b852b",
		Name:         "fw",
		Description:  "OpenStack firewall",
		AdminStateUp: gophercloud.Enabled,
		PolicyID:     "19ab8c87-4a32-4e6a-a74e-b77fffb89a0c",
	}
	_, err := firewalls.Create(fake.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/fw/firewalls/fb5b5315-64f6-4ea3-8e58-981cc37c6f61", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "firewall": {
        "status": "ACTIVE",
        "name": "fw",
        "admin_state_up": true,
        "tenant_id": "b4eedccc6fb74fa8a7ad6b08382b852b",
        "firewall_policy_id": "34be8c83-4d42-4dca-a74e-b77fffb8e28a",
        "id": "fb5b5315-64f6-4ea3-8e58-981cc37c6f61",
        "description": "OpenStack firewall"
    }
}
        `)
	})

	fw, err := firewalls.Get(fake.ServiceClient(), "fb5b5315-64f6-4ea3-8e58-981cc37c6f61").Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "ACTIVE", fw.Status)
	th.AssertEquals(t, "fw", fw.Name)
	th.AssertEquals(t, "OpenStack firewall", fw.Description)
	th.AssertEquals(t, true, fw.AdminStateUp)
	th.AssertEquals(t, "34be8c83-4d42-4dca-a74e-b77fffb8e28a", fw.PolicyID)
	th.AssertEquals(t, "fb5b5315-64f6-4ea3-8e58-981cc37c6f61", fw.ID)
	th.AssertEquals(t, "b4eedccc6fb74fa8a7ad6b08382b852b", fw.TenantID)
}

func TestGetWithExtensions(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/fw/firewalls/fb5b5315-64f6-4ea3-8e58-981cc37c6f61", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "firewall": {
        "status": "ACTIVE",
        "name": "fw",
        "admin_state_up": true,
        "tenant_id": "b4eedccc6fb74fa8a7ad6b08382b852b",
        "firewall_policy_id": "34be8c83-4d42-4dca-a74e-b77fffb8e28a",
        "id": "fb5b5315-64f6-4ea3-8e58-981cc37c6f61",
        "description": "OpenStack firewall",
        "router_ids": ["abcd1234"]
    }
}
        `)
	})

	var fw struct {
		firewalls.Firewall
		routerinsertion.FirewallExt
	}

	err := firewalls.Get(fake.ServiceClient(), "fb5b5315-64f6-4ea3-8e58-981cc37c6f61").ExtractInto(&fw)
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "ACTIVE", fw.Status)
	th.AssertEquals(t, "fw", fw.Name)
	th.AssertEquals(t, "OpenStack firewall", fw.Description)
	th.AssertEquals(t, true, fw.AdminStateUp)
	th.AssertEquals(t, "34be8c83-4d42-4dca-a74e-b77fffb8e28a", fw.PolicyID)
	th.AssertEquals(t, "fb5b5315-64f6-4ea3-8e58-981cc37c6f61", fw.ID)
	th.AssertEquals(t, "b4eedccc6fb74fa8a7ad6b08382b852b", fw.TenantID)
	th.AssertEquals(t, "abcd1234", fw.RouterIDs[0])
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
        "firewall_policy_id": "19ab8c87-4a32-4e6a-a74e-b77fffb89a0c"
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

	options := firewalls.UpdateOpts{
		Name:         "fw",
		Description:  "updated fw",
		AdminStateUp: gophercloud.Disabled,
		PolicyID:     "19ab8c87-4a32-4e6a-a74e-b77fffb89a0c",
	}

	_, err := firewalls.Update(fake.ServiceClient(), "ea5b5315-64f6-4ea3-8e58-981cc37c6576", options).Extract()
	th.AssertNoErr(t, err)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/fw/firewalls/4ec89087-d057-4e2c-911f-60a3b47ee304", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})

	res := firewalls.Delete(fake.ServiceClient(), "4ec89087-d057-4e2c-911f-60a3b47ee304")
	th.AssertNoErr(t, res.Err)
}
