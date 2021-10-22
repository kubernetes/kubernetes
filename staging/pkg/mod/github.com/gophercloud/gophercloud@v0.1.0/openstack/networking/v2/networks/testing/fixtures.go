package testing

import (
	"github.com/gophercloud/gophercloud/openstack/networking/v2/networks"
)

const ListResponse = `
{
    "networks": [
        {
            "status": "ACTIVE",
            "subnets": [
                "54d6f61d-db07-451c-9ab3-b9609b6b6f0b"
            ],
            "name": "public",
            "admin_state_up": true,
            "tenant_id": "4fd44f30292945e481c7b8a0c8908869",
            "shared": true,
            "id": "d32019d3-bc6e-4319-9c1d-6722fc136a22",
            "provider:segmentation_id": 9876543210,
            "provider:physical_network": null,
            "provider:network_type": "local",
            "router:external": true,
            "port_security_enabled": true,
            "dns_domain": "local.",
            "mtu": 1500
        },
        {
            "status": "ACTIVE",
            "subnets": [
                "08eae331-0402-425a-923c-34f7cfe39c1b"
            ],
            "name": "private",
            "admin_state_up": true,
            "tenant_id": "26a7980765d0414dbc1fc1f88cdb7e6e",
            "shared": false,
            "id": "db193ab3-96e3-4cb3-8fc5-05f4296d0324",
            "provider:segmentation_id": 1234567890,
            "provider:physical_network": null,
            "provider:network_type": "local",
            "router:external": false,
            "port_security_enabled": false,
            "dns_domain": "",
            "mtu": 1500
        }
    ]
}`

const GetResponse = `
{
    "network": {
        "status": "ACTIVE",
        "subnets": [
            "54d6f61d-db07-451c-9ab3-b9609b6b6f0b"
        ],
        "name": "public",
        "admin_state_up": true,
        "tenant_id": "4fd44f30292945e481c7b8a0c8908869",
        "shared": true,
        "id": "d32019d3-bc6e-4319-9c1d-6722fc136a22",
        "provider:segmentation_id": 9876543210,
        "provider:physical_network": null,
        "provider:network_type": "local",
        "router:external": true,
        "port_security_enabled": true,
        "dns_domain": "local.",
        "mtu": 1500
    }
}`

const CreateRequest = `
{
    "network": {
        "name": "private",
        "admin_state_up": true
    }
}`

const CreateResponse = `
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
        "provider:network_type": "local",
        "dns_domain": ""
    }
}`

const CreatePortSecurityRequest = `
{
    "network": {
        "name": "private",
        "admin_state_up": true,
        "port_security_enabled": false
    }
}`

const CreatePortSecurityResponse = `
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
        "provider:network_type": "local",
        "port_security_enabled": false
    }
}`

const CreateOptionalFieldsRequest = `
{
  "network": {
      "name": "public",
      "admin_state_up": true,
      "shared": true,
      "tenant_id": "12345",
      "availability_zone_hints": ["zone1", "zone2"]
  }
}`

const UpdateRequest = `
{
    "network": {
        "name": "new_network_name",
        "admin_state_up": false,
        "shared": true
    }
}`

const UpdateResponse = `
{
    "network": {
        "status": "ACTIVE",
        "subnets": [],
        "name": "new_network_name",
        "admin_state_up": false,
        "tenant_id": "4fd44f30292945e481c7b8a0c8908869",
        "shared": true,
        "id": "4e8e5957-649f-477b-9e5b-f1f75b21c03c",
        "provider:segmentation_id": 1234567890,
        "provider:physical_network": null,
        "provider:network_type": "local"
    }
}`

const UpdatePortSecurityRequest = `
{
    "network": {
        "port_security_enabled": false
    }
}`

const UpdatePortSecurityResponse = `
{
    "network": {
        "status": "ACTIVE",
        "subnets": ["08eae331-0402-425a-923c-34f7cfe39c1b"],
        "name": "private",
        "admin_state_up": true,
        "tenant_id": "26a7980765d0414dbc1fc1f88cdb7e6e",
        "shared": false,
        "id": "4e8e5957-649f-477b-9e5b-f1f75b21c03c",
        "provider:segmentation_id": 9876543210,
        "provider:physical_network": null,
        "provider:network_type": "local",
        "port_security_enabled": false
    }
}`

var Network1 = networks.Network{
	Status:       "ACTIVE",
	Subnets:      []string{"54d6f61d-db07-451c-9ab3-b9609b6b6f0b"},
	Name:         "public",
	AdminStateUp: true,
	TenantID:     "4fd44f30292945e481c7b8a0c8908869",
	Shared:       true,
	ID:           "d32019d3-bc6e-4319-9c1d-6722fc136a22",
}

var Network2 = networks.Network{
	Status:       "ACTIVE",
	Subnets:      []string{"08eae331-0402-425a-923c-34f7cfe39c1b"},
	Name:         "private",
	AdminStateUp: true,
	TenantID:     "26a7980765d0414dbc1fc1f88cdb7e6e",
	Shared:       false,
	ID:           "db193ab3-96e3-4cb3-8fc5-05f4296d0324",
}

var ExpectedNetworkSlice = []networks.Network{Network1, Network2}
