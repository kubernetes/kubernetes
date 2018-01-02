package testing

import (
	"fmt"
	"net/http"
	"testing"

	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/layer3/floatingips"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/floatingips", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "floatingips": [
        {
            "floating_network_id": "6d67c30a-ddb4-49a1-bec3-a65b286b4170",
            "router_id": null,
            "fixed_ip_address": null,
            "floating_ip_address": "192.0.0.4",
            "tenant_id": "017d8de156df4177889f31a9bd6edc00",
            "status": "DOWN",
            "port_id": null,
            "id": "2f95fd2b-9f6a-4e8e-9e9a-2cbe286cbf9e",
            "router_id": "1117c30a-ddb4-49a1-bec3-a65b286b4170"
        },
        {
            "floating_network_id": "90f742b1-6d17-487b-ba95-71881dbc0b64",
            "router_id": "0a24cb83-faf5-4d7f-b723-3144ed8a2167",
            "fixed_ip_address": "192.0.0.2",
            "floating_ip_address": "10.0.0.3",
            "tenant_id": "017d8de156df4177889f31a9bd6edc00",
            "status": "DOWN",
            "port_id": "74a342ce-8e07-4e91-880c-9f834b68fa25",
            "id": "ada25a95-f321-4f59-b0e0-f3a970dd3d63",
            "router_id": "2227c30a-ddb4-49a1-bec3-a65b286b4170"
        }
    ]
}
			`)
	})

	count := 0

	floatingips.List(fake.ServiceClient(), floatingips.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := floatingips.ExtractFloatingIPs(page)
		if err != nil {
			t.Errorf("Failed to extract floating IPs: %v", err)
			return false, err
		}

		expected := []floatingips.FloatingIP{
			{
				FloatingNetworkID: "6d67c30a-ddb4-49a1-bec3-a65b286b4170",
				FixedIP:           "",
				FloatingIP:        "192.0.0.4",
				TenantID:          "017d8de156df4177889f31a9bd6edc00",
				Status:            "DOWN",
				PortID:            "",
				ID:                "2f95fd2b-9f6a-4e8e-9e9a-2cbe286cbf9e",
				RouterID:          "1117c30a-ddb4-49a1-bec3-a65b286b4170",
			},
			{
				FloatingNetworkID: "90f742b1-6d17-487b-ba95-71881dbc0b64",
				FixedIP:           "192.0.0.2",
				FloatingIP:        "10.0.0.3",
				TenantID:          "017d8de156df4177889f31a9bd6edc00",
				Status:            "DOWN",
				PortID:            "74a342ce-8e07-4e91-880c-9f834b68fa25",
				ID:                "ada25a95-f321-4f59-b0e0-f3a970dd3d63",
				RouterID:          "2227c30a-ddb4-49a1-bec3-a65b286b4170",
			},
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestInvalidNextPageURLs(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/floatingips", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{"floatingips": [{}], "floatingips_links": {}}`)
	})

	floatingips.List(fake.ServiceClient(), floatingips.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		floatingips.ExtractFloatingIPs(page)
		return true, nil
	})
}

func TestRequiredFieldsForCreate(t *testing.T) {
	res1 := floatingips.Create(fake.ServiceClient(), floatingips.CreateOpts{FloatingNetworkID: ""})
	if res1.Err == nil {
		t.Fatalf("Expected error, got none")
	}

	res2 := floatingips.Create(fake.ServiceClient(), floatingips.CreateOpts{FloatingNetworkID: "foo", PortID: ""})
	if res2.Err == nil {
		t.Fatalf("Expected error, got none")
	}
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/floatingips", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "floatingip": {
        "floating_network_id": "376da547-b977-4cfe-9cba-275c80debf57",
        "port_id": "ce705c24-c1ef-408a-bda3-7bbd946164ab"
    }
}
			`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `
{
    "floatingip": {
        "router_id": "d23abc8d-2991-4a55-ba98-2aaea84cc72f",
        "tenant_id": "4969c491a3c74ee4af974e6d800c62de",
        "floating_network_id": "376da547-b977-4cfe-9cba-275c80debf57",
        "fixed_ip_address": "10.0.0.3",
        "floating_ip_address": "",
        "port_id": "ce705c24-c1ef-408a-bda3-7bbd946164ab",
        "id": "2f245a7b-796b-4f26-9cf9-9e82d248fda7"
    }
}
		`)
	})

	options := floatingips.CreateOpts{
		FloatingNetworkID: "376da547-b977-4cfe-9cba-275c80debf57",
		PortID:            "ce705c24-c1ef-408a-bda3-7bbd946164ab",
	}

	ip, err := floatingips.Create(fake.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "2f245a7b-796b-4f26-9cf9-9e82d248fda7", ip.ID)
	th.AssertEquals(t, "4969c491a3c74ee4af974e6d800c62de", ip.TenantID)
	th.AssertEquals(t, "376da547-b977-4cfe-9cba-275c80debf57", ip.FloatingNetworkID)
	th.AssertEquals(t, "", ip.FloatingIP)
	th.AssertEquals(t, "ce705c24-c1ef-408a-bda3-7bbd946164ab", ip.PortID)
	th.AssertEquals(t, "10.0.0.3", ip.FixedIP)
}

func TestCreateEmptyPort(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/floatingips", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
			{
				"floatingip": {
					"floating_network_id": "376da547-b977-4cfe-9cba-275c80debf57"
				}
			}
			`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `
				{
					"floatingip": {
						"router_id": "d23abc8d-2991-4a55-ba98-2aaea84cc72f",
						"tenant_id": "4969c491a3c74ee4af974e6d800c62de",
						"floating_network_id": "376da547-b977-4cfe-9cba-275c80debf57",
						"fixed_ip_address": "10.0.0.3",
						"floating_ip_address": "",
						"id": "2f245a7b-796b-4f26-9cf9-9e82d248fda7"
					}
				}
				`)
	})

	options := floatingips.CreateOpts{
		FloatingNetworkID: "376da547-b977-4cfe-9cba-275c80debf57",
	}

	ip, err := floatingips.Create(fake.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "2f245a7b-796b-4f26-9cf9-9e82d248fda7", ip.ID)
	th.AssertEquals(t, "4969c491a3c74ee4af974e6d800c62de", ip.TenantID)
	th.AssertEquals(t, "376da547-b977-4cfe-9cba-275c80debf57", ip.FloatingNetworkID)
	th.AssertEquals(t, "", ip.FloatingIP)
	th.AssertEquals(t, "", ip.PortID)
	th.AssertEquals(t, "10.0.0.3", ip.FixedIP)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/floatingips/2f245a7b-796b-4f26-9cf9-9e82d248fda7", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "floatingip": {
        "floating_network_id": "90f742b1-6d17-487b-ba95-71881dbc0b64",
        "fixed_ip_address": "192.0.0.2",
        "floating_ip_address": "10.0.0.3",
        "tenant_id": "017d8de156df4177889f31a9bd6edc00",
        "status": "DOWN",
        "port_id": "74a342ce-8e07-4e91-880c-9f834b68fa25",
        "id": "2f245a7b-796b-4f26-9cf9-9e82d248fda7",
        "router_id": "1117c30a-ddb4-49a1-bec3-a65b286b4170"
    }
}
      `)
	})

	ip, err := floatingips.Get(fake.ServiceClient(), "2f245a7b-796b-4f26-9cf9-9e82d248fda7").Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "90f742b1-6d17-487b-ba95-71881dbc0b64", ip.FloatingNetworkID)
	th.AssertEquals(t, "10.0.0.3", ip.FloatingIP)
	th.AssertEquals(t, "74a342ce-8e07-4e91-880c-9f834b68fa25", ip.PortID)
	th.AssertEquals(t, "192.0.0.2", ip.FixedIP)
	th.AssertEquals(t, "017d8de156df4177889f31a9bd6edc00", ip.TenantID)
	th.AssertEquals(t, "DOWN", ip.Status)
	th.AssertEquals(t, "2f245a7b-796b-4f26-9cf9-9e82d248fda7", ip.ID)
	th.AssertEquals(t, "1117c30a-ddb4-49a1-bec3-a65b286b4170", ip.RouterID)
}

func TestAssociate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/floatingips/2f245a7b-796b-4f26-9cf9-9e82d248fda7", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
	"floatingip": {
		"port_id": "423abc8d-2991-4a55-ba98-2aaea84cc72e"
	}
}
		`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
	"floatingip": {
			"router_id": "d23abc8d-2991-4a55-ba98-2aaea84cc72f",
			"tenant_id": "4969c491a3c74ee4af974e6d800c62de",
			"floating_network_id": "376da547-b977-4cfe-9cba-275c80debf57",
			"fixed_ip_address": null,
			"floating_ip_address": "172.24.4.228",
			"port_id": "423abc8d-2991-4a55-ba98-2aaea84cc72e",
			"id": "2f245a7b-796b-4f26-9cf9-9e82d248fda7"
	}
}
	`)
	})

	portID := "423abc8d-2991-4a55-ba98-2aaea84cc72e"
	ip, err := floatingips.Update(fake.ServiceClient(), "2f245a7b-796b-4f26-9cf9-9e82d248fda7", floatingips.UpdateOpts{PortID: &portID}).Extract()
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, portID, ip.PortID)
}

func TestDisassociate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/floatingips/2f245a7b-796b-4f26-9cf9-9e82d248fda7", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "floatingip": {
      "port_id": null
    }
}
      `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "floatingip": {
        "router_id": "d23abc8d-2991-4a55-ba98-2aaea84cc72f",
        "tenant_id": "4969c491a3c74ee4af974e6d800c62de",
        "floating_network_id": "376da547-b977-4cfe-9cba-275c80debf57",
        "fixed_ip_address": null,
        "floating_ip_address": "172.24.4.228",
        "port_id": null,
        "id": "2f245a7b-796b-4f26-9cf9-9e82d248fda7"
    }
}
    `)
	})

	ip, err := floatingips.Update(fake.ServiceClient(), "2f245a7b-796b-4f26-9cf9-9e82d248fda7", floatingips.UpdateOpts{PortID: nil}).Extract()
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, "", ip.FixedIP)
	th.AssertDeepEquals(t, "", ip.PortID)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/floatingips/2f245a7b-796b-4f26-9cf9-9e82d248fda7", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})

	res := floatingips.Delete(fake.ServiceClient(), "2f245a7b-796b-4f26-9cf9-9e82d248fda7")
	th.AssertNoErr(t, res.Err)
}
