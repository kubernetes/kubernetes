package v2

import (
	"testing"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/workflow/v2/crontriggers"
	"github.com/gophercloud/gophercloud/openstack/workflow/v2/workflows"
	th "github.com/gophercloud/gophercloud/testhelper"
)

// CreateCronTrigger creates a cron trigger for the given workflow.
func CreateCronTrigger(t *testing.T, client *gophercloud.ServiceClient, workflow *workflows.Workflow) (*crontriggers.CronTrigger, error) {
	crontriggerName := tools.RandomString("crontrigger_", 5)
	t.Logf("Attempting to create cron trigger: %s", crontriggerName)

	firstExecution := time.Now().AddDate(1, 0, 0)
	createOpts := crontriggers.CreateOpts{
		WorkflowID: workflow.ID,
		Name:       crontriggerName,
		Pattern:    "0 0 1 1 *",
		WorkflowInput: map[string]interface{}{
			"msg": "Hello World!",
		},
		FirstExecutionTime: &firstExecution,
	}
	crontrigger, err := crontriggers.Create(client, createOpts).Extract()
	if err != nil {
		return crontrigger, err
	}
	t.Logf("Cron trigger created: %s", crontriggerName)
	th.AssertEquals(t, crontrigger.Name, crontriggerName)
	return crontrigger, nil
}

// DeleteCronTrigger deletes a cron trigger.
func DeleteCronTrigger(t *testing.T, client *gophercloud.ServiceClient, crontrigger *crontriggers.CronTrigger) {
	err := crontriggers.Delete(client, crontrigger.ID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete cron trigger %s: %v", crontrigger.Name, err)
	}

	t.Logf("Deleted crontrigger: %s", crontrigger.Name)
}

// GetCronTrigger gets a cron trigger.
func GetCronTrigger(t *testing.T, client *gophercloud.ServiceClient, id string) (*crontriggers.CronTrigger, error) {
	crontrigger, err := crontriggers.Get(client, id).Extract()
	if err != nil {
		t.Fatalf("Unable to get cron trigger %s: %v", id, err)
	}
	t.Logf("Cron trigger %s get", id)
	return crontrigger, err
}

// ListCronTriggers lists cron triggers.
func ListCronTriggers(t *testing.T, client *gophercloud.ServiceClient, opts crontriggers.ListOptsBuilder) ([]crontriggers.CronTrigger, error) {
	allPages, err := crontriggers.List(client, opts).AllPages()
	if err != nil {
		t.Fatalf("Unable to list cron triggers: %v", err)
	}
	crontriggersList, err := crontriggers.ExtractCronTriggers(allPages)
	if err != nil {
		t.Fatalf("Unable to extract cron triggers: %v", err)
	}
	t.Logf("Cron triggers list found, length: %d", len(crontriggersList))
	return crontriggersList, err
}
