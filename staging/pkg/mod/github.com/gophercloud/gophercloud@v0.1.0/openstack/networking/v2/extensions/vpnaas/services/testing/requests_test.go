package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud"
	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/vpnaas/services"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/vpn/vpnservices", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "vpnservice": {
        "router_id": "66e3b16c-8ce5-40fb-bb49-ab6d8dc3f2aa",
        "name": "vpn",
        "admin_state_up": true,
		"description": "OpenStack VPN service",
		"tenant_id":  "10039663455a446d8ba2cbb058b0f578"
    }
}      `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `
{
    "vpnservice": {
        "router_id": "66e3b16c-8ce5-40fb-bb49-ab6d8dc3f2aa",
        "status": "PENDING_CREATE",
        "name": "vpn",
        "external_v6_ip": "2001:db8::1",
        "admin_state_up": true,
        "subnet_id": null,
        "tenant_id": "10039663455a446d8ba2cbb058b0f578",
        "external_v4_ip": "172.32.1.11",
        "id": "5c561d9d-eaea-45f6-ae3e-08d1a7080828",
        "description": "OpenStack VPN service",
		"project_id": "10039663455a446d8ba2cbb058b0f578"
    }
}
    `)
	})

	options := services.CreateOpts{
		TenantID:     "10039663455a446d8ba2cbb058b0f578",
		Name:         "vpn",
		Description:  "OpenStack VPN service",
		AdminStateUp: gophercloud.Enabled,
		RouterID:     "66e3b16c-8ce5-40fb-bb49-ab6d8dc3f2aa",
	}
	actual, err := services.Create(fake.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)
	expected := services.Service{
		RouterID:     "66e3b16c-8ce5-40fb-bb49-ab6d8dc3f2aa",
		Status:       "PENDING_CREATE",
		Name:         "vpn",
		ExternalV6IP: "2001:db8::1",
		AdminStateUp: true,
		SubnetID:     "",
		TenantID:     "10039663455a446d8ba2cbb058b0f578",
		ProjectID:    "10039663455a446d8ba2cbb058b0f578",
		ExternalV4IP: "172.32.1.11",
		ID:           "5c561d9d-eaea-45f6-ae3e-08d1a7080828",
		Description:  "OpenStack VPN service",
	}
	th.AssertDeepEquals(t, expected, *actual)
}

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/vpn/vpnservices", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
   "vpnservices":[
        {
            "router_id": "66e3b16c-8ce5-40fb-bb49-ab6d8dc3f2aa",
            "status": "PENDING_CREATE",
            "name": "vpnservice1",
            "admin_state_up": true,
            "subnet_id": null,
            "project_id": "10039663455a446d8ba2cbb058b0f578",
            "tenant_id": "10039663455a446d8ba2cbb058b0f578",
            "description": "Test VPN service"
        }
   ]
}
      `)
	})

	count := 0

	services.List(fake.ServiceClient(), services.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := services.ExtractServices(page)
		if err != nil {
			t.Errorf("Failed to extract members: %v", err)
			return false, err
		}

		expected := []services.Service{
			{
				Status:       "PENDING_CREATE",
				Name:         "vpnservice1",
				AdminStateUp: true,
				TenantID:     "10039663455a446d8ba2cbb058b0f578",
				ProjectID:    "10039663455a446d8ba2cbb058b0f578",
				Description:  "Test VPN service",
				SubnetID:     "",
				RouterID:     "66e3b16c-8ce5-40fb-bb49-ab6d8dc3f2aa",
			},
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/vpn/vpnservices/5c561d9d-eaea-45f6-ae3e-08d1a7080828", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "vpnservice": {
        "router_id": "66e3b16c-8ce5-40fb-bb49-ab6d8dc3f2aa",
        "status": "PENDING_CREATE",
        "name": "vpnservice1",
        "admin_state_up": true,
        "subnet_id": null,
        "project_id": "10039663455a446d8ba2cbb058b0f578",
        "tenant_id": "10039663455a446d8ba2cbb058b0f578",
        "id": "5c561d9d-eaea-45f6-ae3e-08d1a7080828",
        "description": "VPN test service"
    }
}
        `)
	})

	actual, err := services.Get(fake.ServiceClient(), "5c561d9d-eaea-45f6-ae3e-08d1a7080828").Extract()
	th.AssertNoErr(t, err)
	expected := services.Service{
		Status:       "PENDING_CREATE",
		Name:         "vpnservice1",
		Description:  "VPN test service",
		AdminStateUp: true,
		ID:           "5c561d9d-eaea-45f6-ae3e-08d1a7080828",
		TenantID:     "10039663455a446d8ba2cbb058b0f578",
		ProjectID:    "10039663455a446d8ba2cbb058b0f578",
		RouterID:     "66e3b16c-8ce5-40fb-bb49-ab6d8dc3f2aa",
		SubnetID:     "",
	}
	th.AssertDeepEquals(t, expected, *actual)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/vpn/vpnservices/5c561d9d-eaea-45f6-ae3e-08d1a7080828", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})
	res := services.Delete(fake.ServiceClient(), "5c561d9d-eaea-45f6-ae3e-08d1a7080828")
	th.AssertNoErr(t, res.Err)
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/vpn/vpnservices/5c561d9d-eaea-45f6-ae3e-08d1a7080828", func(w http.ResponseWriter, r *http.Request) {

		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "vpnservice":{
        "name": "updatedname",
        "description": "updated service",
        "admin_state_up": false
    }
}
      `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "vpnservice": {
        "router_id": "66e3b16c-8ce5-40fb-bb49-ab6d8dc3f2aa",
        "status": "PENDING_CREATE",
        "name": "updatedname",
        "admin_state_up": false,
        "subnet_id": null,
        "tenant_id": "10039663455a446d8ba2cbb058b0f578",
        "project_id": "10039663455a446d8ba2cbb058b0f578",
        "id": "5c561d9d-eaea-45f6-ae3e-08d1a7080828",
        "description": "updated service",
		"external_v4_ip": "172.32.1.11",
		"external_v6_ip": "2001:db8::1"
    }
}
    `)
	})
	updatedName := "updatedname"
	updatedServiceDescription := "updated service"
	options := services.UpdateOpts{
		Name:         &updatedName,
		Description:  &updatedServiceDescription,
		AdminStateUp: gophercloud.Disabled,
	}

	actual, err := services.Update(fake.ServiceClient(), "5c561d9d-eaea-45f6-ae3e-08d1a7080828", options).Extract()
	th.AssertNoErr(t, err)
	expected := services.Service{
		RouterID:     "66e3b16c-8ce5-40fb-bb49-ab6d8dc3f2aa",
		Status:       "PENDING_CREATE",
		Name:         "updatedname",
		ExternalV6IP: "2001:db8::1",
		AdminStateUp: false,
		SubnetID:     "",
		TenantID:     "10039663455a446d8ba2cbb058b0f578",
		ProjectID:    "10039663455a446d8ba2cbb058b0f578",
		ExternalV4IP: "172.32.1.11",
		ID:           "5c561d9d-eaea-45f6-ae3e-08d1a7080828",
		Description:  "updated service",
	}
	th.AssertDeepEquals(t, expected, *actual)

}
