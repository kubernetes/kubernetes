package testing

import "github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/layer3/addressscopes"

// AddressScopesListResult represents raw response for the List request.
const AddressScopesListResult = `
{
    "address_scopes": [
        {
            "name": "scopev4",
            "tenant_id": "4a9807b773404e979b19633f38370643",
            "ip_version": 4,
            "shared": false,
            "project_id": "4a9807b773404e979b19633f38370643",
            "id": "9cc35860-522a-4d35-974d-51d4b011801e"
        },
        {
            "name": "scopev6",
            "tenant_id": "4a9807b773404e979b19633f38370643",
            "ip_version": 6,
            "shared": true,
            "project_id": "4a9807b773404e979b19633f38370643",
            "id": "be992b82-bf42-4ab7-bf7b-6baa8759d388"
        }
    ]
}
`

// AddressScope1 represents first unmarshalled address scope from the
// AddressScopesListResult.
var AddressScope1 = addressscopes.AddressScope{
	ID:        "9cc35860-522a-4d35-974d-51d4b011801e",
	Name:      "scopev4",
	TenantID:  "4a9807b773404e979b19633f38370643",
	ProjectID: "4a9807b773404e979b19633f38370643",
	IPVersion: 4,
	Shared:    false,
}

// AddressScope2 represents second unmarshalled address scope from the
// AddressScopesListResult.
var AddressScope2 = addressscopes.AddressScope{
	ID:        "be992b82-bf42-4ab7-bf7b-6baa8759d388",
	Name:      "scopev6",
	TenantID:  "4a9807b773404e979b19633f38370643",
	ProjectID: "4a9807b773404e979b19633f38370643",
	IPVersion: 6,
	Shared:    true,
}

// AddressScopesGetResult represents raw response for the Get request.
const AddressScopesGetResult = `
{
    "address_scope": {
        "name": "scopev4",
        "tenant_id": "4a9807b773404e979b19633f38370643",
        "ip_version": 4,
        "shared": false,
        "project_id": "4a9807b773404e979b19633f38370643",
        "id": "9cc35860-522a-4d35-974d-51d4b011801e"
    }
}
`

// AddressScopeCreateRequest represents raw Create request.
const AddressScopeCreateRequest = `
{
    "address_scope": {
        "ip_version": 4,
        "shared": true,
        "name": "test0"
    }
}
`

// AddressScopeCreateResult represents raw Create response.
const AddressScopeCreateResult = `
{
    "address_scope": {
        "name": "test0",
        "tenant_id": "4a9807b773404e979b19633f38370643",
        "ip_version": 4,
        "shared": true,
        "project_id": "4a9807b773404e979b19633f38370643",
        "id": "9cc35860-522a-4d35-974d-51d4b011801e"
    }
}
`

// AddressScopeUpdateRequest represents raw Update request.
const AddressScopeUpdateRequest = `
{
    "address_scope": {
        "name": "test1",
        "shared": true
    }
}
`

// AddressScopeUpdateResult represents raw Update response.
const AddressScopeUpdateResult = `
{
    "address_scope": {
        "name": "test1",
        "tenant_id": "4a9807b773404e979b19633f38370643",
        "ip_version": 4,
        "shared": true,
        "project_id": "4a9807b773404e979b19633f38370643",
        "id": "9cc35860-522a-4d35-974d-51d4b011801e"
    }
}
`
