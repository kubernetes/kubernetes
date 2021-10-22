// +build acceptance imageservice tasks

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/imageservice/v2/tasks"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestTasksListEachPage(t *testing.T) {
	client, err := clients.NewImageServiceV2Client()
	th.AssertNoErr(t, err)

	listOpts := tasks.ListOpts{
		Limit: 1,
	}

	pager := tasks.List(client, listOpts)
	err = pager.EachPage(func(page pagination.Page) (bool, error) {
		tasks, err := tasks.ExtractTasks(page)
		th.AssertNoErr(t, err)

		for _, task := range tasks {
			tools.PrintResource(t, task)
		}

		return true, nil
	})
}

func TestTasksListAllPages(t *testing.T) {
	client, err := clients.NewImageServiceV2Client()
	th.AssertNoErr(t, err)

	listOpts := tasks.ListOpts{}

	allPages, err := tasks.List(client, listOpts).AllPages()
	th.AssertNoErr(t, err)

	allTasks, err := tasks.ExtractTasks(allPages)
	th.AssertNoErr(t, err)

	for _, i := range allTasks {
		tools.PrintResource(t, i)
	}
}

func TestTaskCreate(t *testing.T) {
	client, err := clients.NewImageServiceV2Client()
	th.AssertNoErr(t, err)

	task, err := CreateTask(t, client, ImportImageURL)
	if err != nil {
		t.Fatalf("Unable to create an Imageservice task: %v", err)
	}

	tools.PrintResource(t, task)
}
