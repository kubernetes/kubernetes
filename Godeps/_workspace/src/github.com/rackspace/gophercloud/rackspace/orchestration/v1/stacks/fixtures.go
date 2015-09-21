package stacks

import (
	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/orchestration/v1/stacks"
)

// CreateExpected represents the expected object from a Create request.
var CreateExpected = &os.CreatedStack{
	ID: "b663e18a-4767-4cdf-9db5-9c8cc13cc38a",
	Links: []gophercloud.Link{
		gophercloud.Link{
			Href: "https://ord.orchestration.api.rackspacecloud.com/v1/864477/stacks/stackcreated/b663e18a-4767-4cdf-9db5-9c8cc13cc38a",
			Rel:  "self",
		},
	},
}

// CreateOutput represents the response body from a Create request.
const CreateOutput = `
{
  "stack": {
    "id": "b663e18a-4767-4cdf-9db5-9c8cc13cc38a",
    "links": [
    {
      "href": "https://ord.orchestration.api.rackspacecloud.com/v1/864477/stacks/stackcreated/b663e18a-4767-4cdf-9db5-9c8cc13cc38a",
      "rel": "self"
    }
    ]
  }
}
`
