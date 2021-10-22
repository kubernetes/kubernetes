package v1

import (
	"fmt"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/orchestration/v1/stacks"
	th "github.com/gophercloud/gophercloud/testhelper"
)

const basicTemplateResourceName = "secgroup_1"
const basicTemplate = `
	{
		"heat_template_version": "2013-05-23",
		"description": "Simple template to test heat commands",
		"resources": {
			"secgroup_1": {
				"type": "OS::Neutron::SecurityGroup",
				"properties": {
					"description": "Gophercloud test",
					"name": "secgroup_1"
				}
			}
		}
	}
`

const validateTemplate = `
	{
		"heat_template_version": "2013-05-23",
		"description": "Simple template to test heat commands",
		"parameters": {
			"flavor": {
				"default": "m1.tiny",
				"type":    "string"
			}
		},
		"resources": {
			"hello_world": {
				"type": "OS::Nova::Server",
				"properties": {
					"key_name": "heat_key",
					"flavor": {
						"get_param": "flavor"
					},
					"image":     "ad091b52-742f-469e-8f3c-fd81cadf0743",
					"user_data": "#!/bin/bash -xv\necho \"hello world\" &gt; /root/hello-world.txt\n"
				}
			}
		}
	}
`

// CreateStack will create a heat stack with a randomly generated name.
// An error will be returned if the stack failed to be created.
func CreateStack(t *testing.T, client *gophercloud.ServiceClient) (*stacks.RetrievedStack, error) {
	stackName := tools.RandomString("ACCPTEST", 8)
	t.Logf("Attempting to create stack %s", stackName)

	template := new(stacks.Template)
	template.Bin = []byte(basicTemplate)

	createOpts := stacks.CreateOpts{
		Name:            stackName,
		Timeout:         60,
		TemplateOpts:    template,
		DisableRollback: gophercloud.Disabled,
	}

	stack, err := stacks.Create(client, createOpts).Extract()
	th.AssertNoErr(t, err)

	if err := WaitForStackStatus(client, stackName, stack.ID, "CREATE_COMPLETE"); err != nil {
		return nil, err
	}

	newStack, err := stacks.Get(client, stackName, stack.ID).Extract()
	return newStack, err
}

// DeleteStack deletes a stack via its ID.
// A fatal error will occur if the stack failed to be deleted. This works
// best when used as a deferred function.
func DeleteStack(t *testing.T, client *gophercloud.ServiceClient, stackName, stackID string) {
	t.Logf("Attempting to delete stack %s (%s)", stackName, stackID)

	err := stacks.Delete(client, stackName, stackID).ExtractErr()
	if err != nil {
		t.Fatalf("Failed to delete stack %s: %s", stackID, err)
	}

	t.Logf("Deleted stack: %s", stackID)
}

// WaitForStackStatus will wait until a stack has reached a certain status.
func WaitForStackStatus(client *gophercloud.ServiceClient, stackName, stackID, status string) error {
	return tools.WaitFor(func() (bool, error) {
		latest, err := stacks.Get(client, stackName, stackID).Extract()
		if err != nil {
			return false, err
		}

		if latest.Status == status {
			return true, nil
		}

		if latest.Status == "ERROR" {
			return false, fmt.Errorf("Stack in ERROR state")
		}

		return false, nil
	})
}
