package routers

import (
	"fmt"
	"net/http"
	"testing"

	fake "github.com/rackspace/gophercloud/openstack/networking/v2/common"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestURLs(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.AssertEquals(t, th.Endpoint()+"v2.0/routers", rootURL(fake.ServiceClient()))
}

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
            "id": "a9254bdb-2613-4a13-ac4c-adc581fba50d"
        }
    ]
}
			`)
	})

	count := 0

	List(fake.ServiceClient(), ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractRouters(page)
		if err != nil {
			t.Errorf("Failed to extract routers: %v", err)
			return false, err
		}

		expected := []Router{
			Router{
				Status:       "ACTIVE",
				GatewayInfo:  GatewayInfo{NetworkID: ""},
				AdminStateUp: true,
				Name:         "second_routers",
				ID:           "7177abc4-5ae9-4bb7-b0d4-89e94a4abf3b",
				TenantID:     "6b96ff0cb17a4b859e1e575d221683d3",
			},
			Router{
				Status:       "ACTIVE",
				GatewayInfo:  GatewayInfo{NetworkID: "3c5bcddd-6af9-4e6b-9c3e-c153e521cab8"},
				AdminStateUp: true,
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
        "id": "8604a0de-7f6b-409a-a47c-a1cc7bc77b2e"
    }
}
		`)
	})

	asu := false
	gwi := GatewayInfo{NetworkID: "8ca37218-28ff-41cb-9b10-039601ea7e6b"}

	options := CreateOpts{
		Name:         "foo_router",
		AdminStateUp: &asu,
		GatewayInfo:  &gwi,
	}
	r, err := Create(fake.ServiceClient(), options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "foo_router", r.Name)
	th.AssertEquals(t, false, r.AdminStateUp)
	th.AssertDeepEquals(t, GatewayInfo{NetworkID: "8ca37218-28ff-41cb-9b10-039601ea7e6b"}, r.GatewayInfo)
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
        "name": "router1",
        "admin_state_up": true,
        "tenant_id": "d6554fe62e2f41efbb6e026fad5c1542",
        "id": "a07eea83-7710-4860-931b-5fe220fae533"
    }
}
			`)
	})

	n, err := Get(fake.ServiceClient(), "a07eea83-7710-4860-931b-5fe220fae533").Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.Status, "ACTIVE")
	th.AssertDeepEquals(t, n.GatewayInfo, GatewayInfo{NetworkID: "85d76829-6415-48ff-9c63-5c5ca8c61ac6"})
	th.AssertEquals(t, n.Name, "router1")
	th.AssertEquals(t, n.AdminStateUp, true)
	th.AssertEquals(t, n.TenantID, "d6554fe62e2f41efbb6e026fad5c1542")
	th.AssertEquals(t, n.ID, "a07eea83-7710-4860-931b-5fe220fae533")
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
        }
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
        "id": "8604a0de-7f6b-409a-a47c-a1cc7bc77b2e"
    }
}
		`)
	})

	gwi := GatewayInfo{NetworkID: "8ca37218-28ff-41cb-9b10-039601ea7e6b"}
	options := UpdateOpts{Name: "new_name", GatewayInfo: &gwi}

	n, err := Update(fake.ServiceClient(), "4e8e5957-649f-477b-9e5b-f1f75b21c03c", options).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, n.Name, "new_name")
	th.AssertDeepEquals(t, n.GatewayInfo, GatewayInfo{NetworkID: "8ca37218-28ff-41cb-9b10-039601ea7e6b"})
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/routers/4e8e5957-649f-477b-9e5b-f1f75b21c03c", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusNoContent)
	})

	res := Delete(fake.ServiceClient(), "4e8e5957-649f-477b-9e5b-f1f75b21c03c")
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

	opts := InterfaceOpts{SubnetID: "a2f1f29d-571b-4533-907f-5803ab96ead1"}
	res, err := AddInterface(fake.ServiceClient(), "4e8e5957-649f-477b-9e5b-f1f75b21c03c", opts).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "0d32a837-8069-4ec3-84c4-3eef3e10b188", res.SubnetID)
	th.AssertEquals(t, "017d8de156df4177889f31a9bd6edc00", res.TenantID)
	th.AssertEquals(t, "3f990102-4485-4df1-97a0-2c35bdb85b31", res.PortID)
	th.AssertEquals(t, "9a83fa11-8da5-436e-9afe-3d3ac5ce7770", res.ID)
}

func TestAddInterfaceRequiredOpts(t *testing.T) {
	_, err := AddInterface(fake.ServiceClient(), "foo", InterfaceOpts{}).Extract()
	if err == nil {
		t.Fatalf("Expected error, got none")
	}
	_, err = AddInterface(fake.ServiceClient(), "foo", InterfaceOpts{SubnetID: "bar", PortID: "baz"}).Extract()
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

	opts := InterfaceOpts{SubnetID: "a2f1f29d-571b-4533-907f-5803ab96ead1"}
	res, err := RemoveInterface(fake.ServiceClient(), "4e8e5957-649f-477b-9e5b-f1f75b21c03c", opts).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, "0d32a837-8069-4ec3-84c4-3eef3e10b188", res.SubnetID)
	th.AssertEquals(t, "017d8de156df4177889f31a9bd6edc00", res.TenantID)
	th.AssertEquals(t, "3f990102-4485-4df1-97a0-2c35bdb85b31", res.PortID)
	th.AssertEquals(t, "9a83fa11-8da5-436e-9afe-3d3ac5ce7770", res.ID)
}
