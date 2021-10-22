package v2

import (
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/workflow/v2/workflows"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestWorkflowsCreateGetDelete(t *testing.T) {
	client, err := clients.NewWorkflowV2Client()
	th.AssertNoErr(t, err)

	workflow, err := CreateWorkflow(t, client)
	th.AssertNoErr(t, err)
	defer DeleteWorkflow(t, client, workflow)

	workflowget, err := GetWorkflow(t, client, workflow.ID)
	th.AssertNoErr(t, err)

	tools.PrintResource(t, workflowget)
}

func TestWorkflowsList(t *testing.T) {
	client, err := clients.NewWorkflowV2Client()
	th.AssertNoErr(t, err)
	workflow, err := CreateWorkflow(t, client)
	th.AssertNoErr(t, err)
	defer DeleteWorkflow(t, client, workflow)
	list, err := ListWorkflows(t, client, &workflows.ListOpts{
		Name: &workflows.ListFilter{
			Value: workflow.Name,
		},
		Tags: []string{"tag1"},
		CreatedAt: &workflows.ListDateFilter{
			Filter: workflows.FilterGT,
			Value:  time.Now().AddDate(-1, 0, 0),
		},
	})
	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, len(list))
	tools.PrintResource(t, list)
}
