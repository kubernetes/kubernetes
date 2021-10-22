package v2

import (
	"fmt"
	"strings"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/workflow/v2/workflows"
	th "github.com/gophercloud/gophercloud/testhelper"
)

// GetEchoWorkflowDefinition returns a simple workflow definition that does nothing except a simple "echo" command.
func GetEchoWorkflowDefinition(workflowName string) string {
	return fmt.Sprintf(`---
version: '2.0'

%s:
  description: Simple workflow example
  type: direct
  tags:
    - tag1
    - tag2

  input:
    - msg

  tasks:
    test:
      action: std.echo output="<%% $.msg %%>"`, workflowName)
}

// CreateWorkflow creates a workflow on Mistral API.
// The created workflow is a dummy workflow that performs a simple echo.
func CreateWorkflow(t *testing.T, client *gophercloud.ServiceClient) (*workflows.Workflow, error) {
	workflowName := tools.RandomString("workflow_echo_", 5)

	definition := GetEchoWorkflowDefinition(workflowName)

	t.Logf("Attempting to create workflow: %s", workflowName)

	opts := &workflows.CreateOpts{
		Namespace:  "some-namespace",
		Scope:      "private",
		Definition: strings.NewReader(definition),
	}
	workflowList, err := workflows.Create(client, opts).Extract()
	if err != nil {
		return nil, err
	}
	th.AssertEquals(t, 1, len(workflowList))

	workflow := workflowList[0]

	t.Logf("Workflow created: %s", workflowName)

	th.AssertEquals(t, workflowName, workflow.Name)

	return &workflow, nil
}

// DeleteWorkflow deletes the given workflow.
func DeleteWorkflow(t *testing.T, client *gophercloud.ServiceClient, workflow *workflows.Workflow) {
	err := workflows.Delete(client, workflow.ID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete workflows %s: %v", workflow.Name, err)
	}

	t.Logf("Deleted workflow: %s", workflow.Name)
}

// GetWorkflow gets a workflow.
func GetWorkflow(t *testing.T, client *gophercloud.ServiceClient, id string) (*workflows.Workflow, error) {
	workflow, err := workflows.Get(client, id).Extract()
	if err != nil {
		t.Fatalf("Unable to get workflow %s: %v", id, err)
	}
	t.Logf("Workflow get: %s", workflow.Name)
	return workflow, err
}

// ListWorkflows lists the workflows.
func ListWorkflows(t *testing.T, client *gophercloud.ServiceClient, opts workflows.ListOptsBuilder) ([]workflows.Workflow, error) {
	allPages, err := workflows.List(client, opts).AllPages()
	if err != nil {
		t.Fatalf("Unable to list workflows: %v", err)
	}
	workflowsList, err := workflows.ExtractWorkflows(allPages)
	if err != nil {
		t.Fatalf("Unable to extract workflows: %v", err)
	}
	t.Logf("Workflows list find, length: %d", len(workflowsList))
	return workflowsList, err
}
