package testing

import (
	"fmt"
	"net/http"
	"testing"

	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	floatingiptest "github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/layer3/floatingips/testing"
	networktest "github.com/gophercloud/gophercloud/openstack/networking/v2/networks/testing"
	porttest "github.com/gophercloud/gophercloud/openstack/networking/v2/ports/testing"
	th "github.com/gophercloud/gophercloud/testhelper"
)

const NetworkCreateRequest = `
{
    "network": {
        "name": "private",
        "admin_state_up": true,
        "dns_domain": "local."
    }
}`

const NetworkCreateResponse = `
{
    "network": {
        "status": "ACTIVE",
        "subnets": ["08eae331-0402-425a-923c-34f7cfe39c1b"],
        "name": "private",
        "admin_state_up": true,
        "tenant_id": "26a7980765d0414dbc1fc1f88cdb7e6e",
        "shared": false,
        "id": "db193ab3-96e3-4cb3-8fc5-05f4296d0324",
        "provider:segmentation_id": 9876543210,
        "provider:physical_network": null,
        "provider:network_type": "local.",
        "dns_domain": "local."
    }
}`

const NetworkUpdateRequest = `
{
    "network": {
        "name": "new_network_name",
        "admin_state_up": false,
        "dns_domain": ""
    }
}`

const NetworkUpdateResponse = `
{
    "network": {
        "status": "ACTIVE",
        "subnets": ["08eae331-0402-425a-923c-34f7cfe39c1b"],
        "name": "new_network_name",
        "admin_state_up": false,
        "tenant_id": "26a7980765d0414dbc1fc1f88cdb7e6e",
        "shared": false,
        "id": "db193ab3-96e3-4cb3-8fc5-05f4296d0324",
        "provider:segmentation_id": 9876543210,
        "provider:physical_network": null,
        "provider:network_type": "local.",
        "dns_domain": ""
    }
}`

func PortHandleListSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/ports", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		th.AssertEquals(t, r.RequestURI, "/v2.0/ports?dns_name=test-port")

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, porttest.ListResponse)
	})
}

func PortHandleGet(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/ports/46d4bfb9-b26e-41f3-bd2e-e6dcc1ccedb2", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, porttest.GetResponse)
	})
}

func PortHandleCreate(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/ports", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "port": {
        "network_id": "a87cc70a-3e15-4acf-8205-9b711a3531b7",
        "name": "private-port",
        "admin_state_up": true,
        "fixed_ips": [
            {
                "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
                "ip_address": "10.0.0.2"
            }
        ],
        "security_groups": ["foo"],
        "dns_name": "test-port"
    }
}
      `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, `
{
    "port": {
        "status": "DOWN",
        "name": "private-port",
        "allowed_address_pairs": [],
        "admin_state_up": true,
        "network_id": "a87cc70a-3e15-4acf-8205-9b711a3531b7",
        "tenant_id": "d6700c0c9ffa4f1cb322cd4a1f3906fa",
        "device_owner": "",
        "mac_address": "fa:16:3e:c9:cb:f0",
        "fixed_ips": [
            {
                "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
                "ip_address": "10.0.0.2"
            }
        ],
        "dns_name": "test-port",
        "dns_assignment": [
          {
            "hostname": "test-port",
            "ip_address": "172.24.4.2",
            "fqdn": "test-port.openstack.local."
          }
        ],
        "id": "65c0ee9f-d634-4522-8954-51021b570b0d",
        "security_groups": [
            "f0ac4394-7e4a-4409-9701-ba8be283dbc3"
        ],
        "device_id": ""
    }
}
    `)
	})
}

func PortHandleUpdate(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/ports/65c0ee9f-d634-4522-8954-51021b570b0d", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "port": {
      "name": "new_port_name",
      "fixed_ips": [
          {
              "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
              "ip_address": "10.0.0.3"
          }
      ],
      "security_groups": [
        "f0ac4394-7e4a-4409-9701-ba8be283dbc3"
      ],
      "dns_name": "test-port1"
    }
}
      `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
    "port": {
        "status": "DOWN",
        "name": "new_port_name",
        "admin_state_up": true,
        "network_id": "a87cc70a-3e15-4acf-8205-9b711a3531b7",
        "tenant_id": "d6700c0c9ffa4f1cb322cd4a1f3906fa",
        "device_owner": "",
        "mac_address": "fa:16:3e:c9:cb:f0",
        "fixed_ips": [
            {
                "subnet_id": "a0304c3a-4f08-4c43-88af-d796509c97d2",
                "ip_address": "10.0.0.3"
            }
        ],
        "id": "65c0ee9f-d634-4522-8954-51021b570b0d",
        "security_groups": [
            "f0ac4394-7e4a-4409-9701-ba8be283dbc3"
        ],
        "device_id": "",
        "dns_name": "test-port1",
        "dns_assignment": [
          {
            "hostname": "test-port1",
            "ip_address": "172.24.4.2",
            "fqdn": "test-port1.openstack.local."
          }
        ]
    }
}
    `)
	})
}

func FloatingIPHandleList(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/floatingips", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		th.AssertEquals(t, r.RequestURI, "/v2.0/floatingips?dns_domain=local.&dns_name=test-fip")

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, floatingiptest.ListResponseDNS)
	})
}

func FloatingIPHandleGet(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/floatingips/2f95fd2b-9f6a-4e8e-9e9a-2cbe286cbf9e", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, fmt.Sprintf(`{"floatingip": %s}`, floatingiptest.FipDNS))
	})
}

func FloatingIPHandleCreate(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/floatingips", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
{
    "floatingip": {
        "floating_network_id": "6d67c30a-ddb4-49a1-bec3-a65b286b4170",
        "dns_name": "test-fip",
        "dns_domain": "local."
    }
}
                        `)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, fmt.Sprintf(`{"floatingip": %s}`, floatingiptest.FipDNS))
	})
}

func NetworkHandleList(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/networks", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		th.AssertEquals(t, r.RequestURI, "/v2.0/networks?dns_domain=local.")

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, networktest.ListResponse)
	})
}

func NetworkHandleGet(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/networks/d32019d3-bc6e-4319-9c1d-6722fc136a22", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, networktest.GetResponse)
	})
}

func NetworkHandleCreate(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/networks", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, NetworkCreateRequest)
		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprintf(w, NetworkCreateResponse)
	})
}

func NetworkHandleUpdate(t *testing.T) {
	th.Mux.HandleFunc("/v2.0/networks/db193ab3-96e3-4cb3-8fc5-05f4296d0324", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, NetworkUpdateRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, NetworkUpdateResponse)
	})
}
