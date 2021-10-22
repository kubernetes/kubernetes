package testing

// NetworksVLANTransparentListResult represents raw HTTP response for the List
// request.
const NetworksVLANTransparentListResult = `
{
    "networks": [
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
            "vlan_transparent": true
        },
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
            "port_security_enabled": true
        }
    ]
}`

// NetworksVLANTransparentGetResult represents raw HTTP response for the Get
// request.
const NetworksVLANTransparentGetResult = `
{
    "network": {
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
        "vlan_transparent": true
    }
}`

// NetworksVLANTransparentCreateRequest represents raw HTTP Create request.
const NetworksVLANTransparentCreateRequest = `
{
    "network": {
        "name": "private",
        "admin_state_up": true,
        "vlan_transparent": true
    }
}`

// NetworksVLANTransparentCreateResult represents raw HTTP response for the
// Create request.
const NetworksVLANTransparentCreateResult = `
{
    "network": {
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
        "vlan_transparent": true
    }
}
`

// NetworksVLANTransparentUpdateRequest represents raw HTTP Update request.
const NetworksVLANTransparentUpdateRequest = `
{
    "network": {
        "name": "new_network_name",
        "admin_state_up": false,
        "vlan_transparent": false
    }
}`

// NetworksVLANTransparentUpdateResult represents raw HTTP response for the
// Update request.
const NetworksVLANTransparentUpdateResult = `
{
    "network": {
        "status": "ACTIVE",
        "subnets": [
            "08eae331-0402-425a-923c-34f7cfe39c1b"
        ],
        "name": "new_network_name",
        "admin_state_up": false,
        "tenant_id": "26a7980765d0414dbc1fc1f88cdb7e6e",
        "shared": false,
        "id": "db193ab3-96e3-4cb3-8fc5-05f4296d0324",
        "provider:segmentation_id": 1234567890,
        "provider:physical_network": null,
        "provider:network_type": "local",
        "router:external": false,
        "port_security_enabled": false,
        "vlan_transparent": false
    }
}
`
