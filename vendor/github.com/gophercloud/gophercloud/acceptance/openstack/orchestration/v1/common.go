// +build acceptance

package v1

import (
	"fmt"
	"os"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack"
	th "github.com/gophercloud/gophercloud/testhelper"
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
}`, os.Getenv("OS_FLAVOR_ID"), os.Getenv("OS_IMAGE_ID"))

func newClient(t *testing.T) *gophercloud.ServiceClient {
	ao, err := openstack.AuthOptionsFromEnv()
	th.AssertNoErr(t, err)

	client, err := openstack.AuthenticatedClient(ao)
	th.AssertNoErr(t, err)

	c, err := openstack.NewOrchestrationV1(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
	th.AssertNoErr(t, err)
	return c
}
