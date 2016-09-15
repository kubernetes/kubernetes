package testing

import (
	"fmt"
	"net/http"
	"testing"

	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/layer3/routers"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/routers", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "routers": [
        {
            "status": "ACTIVE",
            "external_gateway_info": null,
            "name": "second_routers",
            "admin_state_up": true,
            "tenant_id": "6b96ff0cb17a4b859e1e575d221683d3",
            "distributed": false,
            "id": "7177abc4-5ae9-4bb7-b0d4-89e94a4abf3b"
        },
        {
            "status": "ACTIVE",
            "external_gateway_info": {
                "network_id": "3c5bcddd-6af9-4e6b-9c3e-c153e521cab8"
            },
            "name": "router1",
            "admin_state_up": true,
            "tenant_id": "33a40233088643acb66ff6eb0ebea679",
            "distributed": false,
            "id": "a9254bdb-2613-4a13-ac4c-adc581fba50d"
        }
    ]
}
			`)
	})

	count := 0

	routers.List(fake.ServiceClient(), routers.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := routers.ExtractRouters(page)
		if err != nil {
			t.Errorf("Failed to extract routers: %v", err)
			return false, err
		}

		expected := []routers.Router{
			{
				Status:       "ACTIVE",
				GatewayInfo:  routers.GatewayInfo{NetworkID: ""},
				AdminStateUp: true,
				Distributed:  false,
				Name:         "second_routers",
				ID:           "7177abc4-5ae9-4bb7-b0d4-89e94a4abf3b",
				TenantID:     "6b96ff0cb17a4b859e1e575d221683d3",
			},
			{
				Status:       "ACTIVE",
				GatewayInfo:  routers.GatewayInfo{NetworkID: "3c5bcddd-6af9-4e6b-9c3e-c153e521cab8"},
				AdminStateUp: true,
				Distributed:  false,
				Name:         "router1",
				ID:           "a9254bdb-2613-4a13-ac4c-adc581fba50d",
				TenantID:     "33a40233088643acb66ff6eb0ebea679",
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

	th.Mux.HandleFunc("/v2.0/routers", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
   "router":{
      "name": "foo_router",
      "admin_state_up": false,
      "external_gateway_info":{
         "network_id":"8ca37218-28ff-41cb-9b10-039601ea7e6b"
      }
   }
}
			`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `
{
    "router": {
        "status": "ACTIVE",
        "external_gateway_info": {
            "network_id": "8ca37218-28ff-41cb-9b10-039601ea7e6b"
        },
        "name": "foo_router",
        "admin_state_up": false,
        "tenant_id": "6b96ff0cb17a4b859e1e575d221683d3",
        "distributed": false,
        "id": "8604a0de-7f6b-409a-a47c-a1cc7bc77b2e"
    }
}
		`)
	})

	asu := false
	gwi := routers.GatewayInfo{NetworkID: "8ca37218-28ff-41cb-9b10-039601ea7e6b"}

	options := routers.CreateOpts{
		Name:         "foo_router",
		AdminStateUp: &asu,
		GatewayInfo:  &gwi,
	}
	r, err := routers.Create(fake.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "foo_router", r.Name)
	th.AssertEquals(t, false, r.AdminStateUp)
	th.AssertDeepEquals(t, routers.GatewayInfo{NetworkID: "8ca37218-28ff-41cb-9b10-039601ea7e6b"}, r.GatewayInfo)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/routers/a07eea83-7710-4860-931b-5fe220fae533", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "router": {
        "status": "ACTIVE",
        "external_gateway_info": {
            "network_id": "85d76829-6415-48ff-9c63-5c5ca8c61ac6"
        },
        "routes": [
            {
                "nexthop": "10.1.0.10",
                "destination": "40.0.1.0/24"
            }
        ],
        "name": "router1",
        "admin_state_up": true,
        "tenant_id": "d6554fe62e2f41efbb6e026fad5c1542",
        "distributed": false,
        "id": "a07eea83-7710-4860-931b-5fe220fae533"
    }
}
			`)
	})

	n, err := routers.Get(fake.ServiceClient(), "a07eea83-7710-4860-931b-5fe220fae533").Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.Status, "ACTIVE")
	th.AssertDeepEquals(t, n.GatewayInfo, routers.GatewayInfo{NetworkID: "85d76829-6415-48ff-9c63-5c5ca8c61ac6"})
	th.AssertEquals(t, n.Name, "router1")
	th.AssertEquals(t, n.AdminStateUp, true)
	th.AssertEquals(t, n.TenantID, "d6554fe62e2f41efbb6e026fad5c1542")
	th.AssertEquals(t, n.ID, "a07eea83-7710-4860-931b-5fe220fae533")
	th.AssertDeepEquals(t, n.Routes, []routers.Route{{DestinationCIDR: "40.0.1.0/24", NextHop: "10.1.0.10"}})
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/routers/4e8e5957-649f-477b-9e5b-f1f75b21c03c", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "router": {
			"name": "new_name",
        "external_gateway_info": {
            "network_id": "8ca37218-28ff-41cb-9b10-039601ea7e6b"
		},
        "routes": [
            {
                "nexthop": "10.1.0.10",
                "destination": "40.0.1.0/24"
            }
        ]
    }
}
			`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "router": {
        "status": "ACTIVE",
        "external_gateway_info": {
            "network_id": "8ca37218-28ff-41cb-9b10-039601ea7e6b"
        },
        "name": "new_name",
        "admin_state_up": true,
        "tenant_id": "6b96ff0cb17a4b859e1e575d221683d3",
        "distributed": false,
        "id": "8604a0de-7f6b-409a-a47c-a1cc7bc77b2e",
        "routes": [
            {
                "nexthop": "10.1.0.10",
                "destination": "40.0.1.0/24"
            }
        ]
    }
}
		`)
	})

	gwi := routers.GatewayInfo{NetworkID: "8ca37218-28ff-41cb-9b10-039601ea7e6b"}
	r := []routers.Route{{DestinationCIDR: "40.0.1.0/24", NextHop: "10.1.0.10"}}
	options := routers.UpdateOpts{Name: "new_name", GatewayInfo: &gwi, Routes: r}

	n, err := routers.Update(fake.ServiceClient(), "4e8e5957-649f-477b-9e5b-f1f75b21c03c", options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.Name, "new_name")
	th.AssertDeepEquals(t, n.GatewayInfo, routers.GatewayInfo{NetworkID: "8ca37218-28ff-41cb-9b10-039601ea7e6b"})
	th.AssertDeepEquals(t, n.Routes, []routers.Route{{DestinationCIDR: "40.0.1.0/24", NextHop: "10.1.0.10"}})
}

func TestAllRoutesRemoved(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/routers/4e8e5957-649f-477b-9e5b-f1f75b21c03c", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "router": {
        "routes": []
    }
}
			`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "router": {
        "status": "ACTIVE",
        "external_gateway_info": {
            "network_id": "8ca37218-28ff-41cb-9b10-039601ea7e6b"
        },
        "name": "name",
        "admin_state_up": true,
        "tenant_id": "6b96ff0cb17a4b859e1e575d221683d3",
        "distributed": false,
        "id": "8604a0de-7f6b-409a-a47c-a1cc7bc77b2e",
        "routes": []
    }
}
		`)
	})

	r := []routers.Route{}
	options := routers.UpdateOpts{Routes: r}

	n, err := routers.Update(fake.ServiceClient(), "4e8e5957-649f-477b-9e5b-f1f75b21c03c", options).Extract()
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, n.Routes, []routers.Route{})
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/routers/4e8e5957-649f-477b-9e5b-f1f75b21c03c", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})

	res := routers.Delete(fake.ServiceClient(), "4e8e5957-649f-477b-9e5b-f1f75b21c03c")
	th.AssertNoErr(t, res.Err)
}

func TestAddInterface(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/routers/4e8e5957-649f-477b-9e5b-f1f75b21c03c/add_router_interface", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "subnet_id": "a2f1f29d-571b-4533-907f-5803ab96ead1"
}
	`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "subnet_id": "0d32a837-8069-4ec3-84c4-3eef3e10b188",
    "tenant_id": "017d8de156df4177889f31a9bd6edc00",
    "port_id": "3f990102-4485-4df1-97a0-2c35bdb85b31",
    "id": "9a83fa11-8da5-436e-9afe-3d3ac5ce7770"
}
`)
	})

	opts := routers.AddInterfaceOpts{SubnetID: "a2f1f29d-571b-4533-907f-5803ab96ead1"}
	res, err := routers.AddInterface(fake.ServiceClient(), "4e8e5957-649f-477b-9e5b-f1f75b21c03c", opts).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "0d32a837-8069-4ec3-84c4-3eef3e10b188", res.SubnetID)
	th.AssertEquals(t, "017d8de156df4177889f31a9bd6edc00", res.TenantID)
	th.AssertEquals(t, "3f990102-4485-4df1-97a0-2c35bdb85b31", res.PortID)
	th.AssertEquals(t, "9a83fa11-8da5-436e-9afe-3d3ac5ce7770", res.ID)
}

func TestAddInterfaceRequiredOpts(t *testing.T) {
	_, err := routers.AddInterface(fake.ServiceClient(), "foo", routers.AddInterfaceOpts{}).Extract()
	if err == nil {
		t.Fatalf("Expected error, got none")
	}
	_, err = routers.AddInterface(fake.ServiceClient(), "foo", routers.AddInterfaceOpts{SubnetID: "bar", PortID: "baz"}).Extract()
	if err == nil {
		t.Fatalf("Expected error, got none")
	}
}

func TestRemoveInterface(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/routers/4e8e5957-649f-477b-9e5b-f1f75b21c03c/remove_router_interface", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
		"subnet_id": "a2f1f29d-571b-4533-907f-5803ab96ead1"
}
	`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
		"subnet_id": "0d32a837-8069-4ec3-84c4-3eef3e10b188",
		"tenant_id": "017d8de156df4177889f31a9bd6edc00",
		"port_id": "3f990102-4485-4df1-97a0-2c35bdb85b31",
		"id": "9a83fa11-8da5-436e-9afe-3d3ac5ce7770"
}
`)
	})

	opts := routers.RemoveInterfaceOpts{SubnetID: "a2f1f29d-571b-4533-907f-5803ab96ead1"}
	res, err := routers.RemoveInterface(fake.ServiceClient(), "4e8e5957-649f-477b-9e5b-f1f75b21c03c", opts).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "0d32a837-8069-4ec3-84c4-3eef3e10b188", res.SubnetID)
	th.AssertEquals(t, "017d8de156df4177889f31a9bd6edc00", res.TenantID)
	th.AssertEquals(t, "3f990102-4485-4df1-97a0-2c35bdb85b31", res.PortID)
	th.AssertEquals(t, "9a83fa11-8da5-436e-9afe-3d3ac5ce7770", res.ID)
}
