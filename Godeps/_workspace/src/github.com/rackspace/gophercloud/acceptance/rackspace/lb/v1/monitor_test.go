// +build acceptance lbs

package v1

import (
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/rackspace/lb/v1/lbs"
	"github.com/rackspace/gophercloud/rackspace/lb/v1/monitors"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestMonitors(t *testing.T) {
	client := setup(t)

	ids := createLB(t, client, 1)
	lbID := ids[0]

	getMonitor(t, client, lbID)

	updateMonitor(t, client, lbID)

	deleteMonitor(t, client, lbID)

	deleteLB(t, client, lbID)
}

func getMonitor(t *testing.T, client *gophercloud.ServiceClient, lbID int) {
	hm, err := monitors.Get(client, lbID).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Health monitor for LB %d: Type [%s] Delay [%d] Timeout [%d] AttemptLimit [%d]",
		lbID, hm.Type, hm.Delay, hm.Timeout, hm.AttemptLimit)
}

func updateMonitor(t *testing.T, client *gophercloud.ServiceClient, lbID int) {
	opts := monitors.UpdateHTTPMonitorOpts{
		AttemptLimit: 3,
		Delay:        10,
		Timeout:      10,
		BodyRegex:    "hello is it me you're looking for",
		Path:         "/foo",
		StatusRegex:  "200",
		Type:         monitors.HTTP,
	}

	err := monitors.Update(client, lbID, opts).ExtractErr()
	th.AssertNoErr(t, err)

	waitForLB(client, lbID, lbs.ACTIVE)
	t.Logf("Updated monitor for LB %d", lbID)
}

func deleteMonitor(t *testing.T, client *gophercloud.ServiceClient, lbID int) {
	err := monitors.Delete(client, lbID).ExtractErr()
	th.AssertNoErr(t, err)

	waitForLB(client, lbID, lbs.ACTIVE)
	t.Logf("Deleted monitor for LB %d", lbID)
}
