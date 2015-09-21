// +build acceptance networking lbaas lbaasmonitor

package lbaas

import (
	"testing"

	base "github.com/rackspace/gophercloud/acceptance/openstack/networking/v2"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/monitors"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestMonitors(t *testing.T) {
	base.Setup(t)
	defer base.Teardown()

	// create monitor
	monitorID := CreateMonitor(t)

	// list monitors
	listMonitors(t)

	// update monitor
	updateMonitor(t, monitorID)

	// get monitor
	getMonitor(t, monitorID)

	// delete monitor
	deleteMonitor(t, monitorID)
}

func listMonitors(t *testing.T) {
	err := monitors.List(base.Client, monitors.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		monitorList, err := monitors.ExtractMonitors(page)
		if err != nil {
			t.Errorf("Failed to extract monitors: %v", err)
			return false, err
		}

		for _, m := range monitorList {
			t.Logf("Listing monitor: ID [%s] Type [%s] Delay [%ds] Timeout [%d] Retries [%d] Status [%s]",
				m.ID, m.Type, m.Delay, m.Timeout, m.MaxRetries, m.Status)
		}

		return true, nil
	})

	th.AssertNoErr(t, err)
}

func updateMonitor(t *testing.T, monitorID string) {
	opts := monitors.UpdateOpts{Delay: 10, Timeout: 10, MaxRetries: 3}
	m, err := monitors.Update(base.Client, monitorID, opts).Extract()

	th.AssertNoErr(t, err)

	t.Logf("Updated monitor ID [%s]", m.ID)
}

func getMonitor(t *testing.T, monitorID string) {
	m, err := monitors.Get(base.Client, monitorID).Extract()

	th.AssertNoErr(t, err)

	t.Logf("Getting monitor ID [%s]: URL path [%s] HTTP Method [%s] Accepted codes [%s]",
		m.ID, m.URLPath, m.HTTPMethod, m.ExpectedCodes)
}

func deleteMonitor(t *testing.T, monitorID string) {
	res := monitors.Delete(base.Client, monitorID)

	th.AssertNoErr(t, res.Err)

	t.Logf("Deleted monitor %s", monitorID)
}
