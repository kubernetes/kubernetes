// +build acceptance

package v1

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/orchestration/v1/stacktemplates"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestStackTemplatesCRUD(t *testing.T) {
	client, err := clients.NewOrchestrationV1Client()
	th.AssertNoErr(t, err)

	stack, err := CreateStack(t, client)
	th.AssertNoErr(t, err)
	defer DeleteStack(t, client, stack.Name, stack.ID)

	tmpl, err := stacktemplates.Get(client, stack.Name, stack.ID).Extract()
	th.AssertNoErr(t, err)
	tools.PrintResource(t, tmpl)
}

func TestStackTemplatesValidate(t *testing.T) {
	client, err := clients.NewOrchestrationV1Client()
	th.AssertNoErr(t, err)

	validateOpts := stacktemplates.ValidateOpts{
		Template: validateTemplate,
	}

	validatedTemplate, err := stacktemplates.Validate(client, validateOpts).Extract()
	th.AssertNoErr(t, err)
	tools.PrintResource(t, validatedTemplate)
}
