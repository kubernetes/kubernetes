// +build acceptance

package v1

import (
	"fmt"
	"os"
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/rackspace"
	th "github.com/rackspace/gophercloud/testhelper"
)

var template = fmt.Sprintf(`
{
		"heat_template_version": "2013-05-23",
		"description": "Simple template to test heat commands",
		"parameters": {},
		"resources": {
				"hello_world": {
						"type":"OS::Nova::Server",
						"properties": {
								"flavor": "%s",
								"image": "%s",
								"user_data": "#!/bin/bash -xv\necho \"hello world\" &gt; /root/hello-world.txt\n"
						}
				}
		}
}
`, os.Getenv("RS_FLAVOR_ID"), os.Getenv("RS_IMAGE_ID"))

func newClient(t *testing.T) *gophercloud.ServiceClient {
	ao, err := rackspace.AuthOptionsFromEnv()
	th.AssertNoErr(t, err)

	client, err := rackspace.AuthenticatedClient(ao)
	th.AssertNoErr(t, err)

	c, err := rackspace.NewOrchestrationV1(client, gophercloud.EndpointOpts{
		Region: os.Getenv("RS_REGION_NAME"),
	})
	th.AssertNoErr(t, err)
	return c
}
