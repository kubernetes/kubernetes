package testing

import (
	"encoding/json"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/subnets"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestHostRoute(t *testing.T) {
	sejson := []byte(`
    {"subnet": {
      "name": "test-subnet",
      "enable_dhcp": false,
      "network_id": "3e66c41e-cbbd-4019-9aab-740b7e4150a0",
      "tenant_id": "f86e123198cf42d19c8854c5f80c2f06",
      "dns_nameservers": [],
      "gateway_ip": "172.16.0.1",
      "ipv6_ra_mode": null,
      "allocation_pools": [
        {
          "start": "172.16.0.2",
          "end": "172.16.255.254"
        }
      ],
      "host_routes": [
        {
          "destination": "172.20.1.0/24",
		  		"nexthop": "172.16.0.2"
        }
      ],
      "ip_version": 4,
      "ipv6_address_mode": null,
      "cidr": "172.16.0.0/16",
      "id": "6dcaa873-7115-41af-9ef5-915f73636e43",
      "subnetpool_id": null
  }}
`)

	var dejson interface{}
	err := json.Unmarshal(sejson, &dejson)
	if err != nil {
		t.Fatalf("%s", err)
	}

	resp := gophercloud.Result{Body: dejson}
	var subnetWrapper struct {
		Subnet subnets.Subnet `json:"subnet"`
	}
	err = resp.ExtractInto(&subnetWrapper)
	if err != nil {
		t.Fatalf("%s", err)
	}
	route := subnetWrapper.Subnet.HostRoutes[0]
	th.AssertEquals(t, route.NextHop, "172.16.0.2")
	th.AssertEquals(t, route.DestinationCIDR, "172.20.1.0/24")
}
