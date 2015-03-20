// +build acceptance lbs

package v1

import (
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/rackspace/lb/v1/throttle"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestThrottle(t *testing.T) {
	client := setup(t)

	ids := createLB(t, client, 1)
	lbID := ids[0]

	getThrottleConfig(t, client, lbID)

	createThrottleConfig(t, client, lbID)
	waitForLB(client, lbID, "ACTIVE")

	deleteThrottleConfig(t, client, lbID)
	waitForLB(client, lbID, "ACTIVE")

	deleteLB(t, client, lbID)
}

func getThrottleConfig(t *testing.T, client *gophercloud.ServiceClient, lbID int) {
	sp, err := throttle.Get(client, lbID).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Throttle config: MaxConns [%s]", sp.MaxConnections)
}

func createThrottleConfig(t *testing.T, client *gophercloud.ServiceClient, lbID int) {
	opts := throttle.CreateOpts{
		MaxConnections:    200,
		MaxConnectionRate: 100,
		MinConnections:    0,
		RateInterval:      10,
	}

	err := throttle.Create(client, lbID, opts).ExtractErr()
	th.AssertNoErr(t, err)
	t.Logf("Enable throttling for %d", lbID)
}

func deleteThrottleConfig(t *testing.T, client *gophercloud.ServiceClient, lbID int) {
	err := throttle.Delete(client, lbID).ExtractErr()
	th.AssertNoErr(t, err)
	t.Logf("Disable throttling for %d", lbID)
}
