package testing

// These fixtures are here instead of in the underlying networks package
// because all network tests (including extensions) would have to
// implement the NetworkExternalExt extention for create/update tests
// to pass.

const CreateRequest = `
{
    "network": {
        "name": "private",
        "admin_state_up": true,
        "router:external": false
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
        "router:external": false
    }
}`

const UpdateRequest = `
{
    "network": {
        "name": "new_network_name",
        "admin_state_up": false,
        "shared": true,
        "router:external": false
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
        "provider:network_type": "local",
        "router:external": false
    }
}`

const ExpectedListOpts = "?id=d32019d3-bc6e-4319-9c1d-6722fc136a22&router%3Aexternal=true"
