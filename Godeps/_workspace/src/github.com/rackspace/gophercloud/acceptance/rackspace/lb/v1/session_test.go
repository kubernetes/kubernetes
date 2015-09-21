// +build acceptance lbs

package v1

import (
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/rackspace/lb/v1/sessions"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestSession(t *testing.T) {
	client := setup(t)

	ids := createLB(t, client, 1)
	lbID := ids[0]

	getSession(t, client, lbID)

	enableSession(t, client, lbID)
	waitForLB(client, lbID, "ACTIVE")

	disableSession(t, client, lbID)
	waitForLB(client, lbID, "ACTIVE")

	deleteLB(t, client, lbID)
}

func getSession(t *testing.T, client *gophercloud.ServiceClient, lbID int) {
	sp, err := sessions.Get(client, lbID).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Session config: Type [%s]", sp.Type)
}

func enableSession(t *testing.T, client *gophercloud.ServiceClient, lbID int) {
	opts := sessions.CreateOpts{Type: sessions.HTTPCOOKIE}
	err := sessions.Enable(client, lbID, opts).ExtractErr()
	th.AssertNoErr(t, err)
	t.Logf("Enable %s sessions for %d", opts.Type, lbID)
}

func disableSession(t *testing.T, client *gophercloud.ServiceClient, lbID int) {
	err := sessions.Disable(client, lbID).ExtractErr()
	th.AssertNoErr(t, err)
	t.Logf("Disable sessions for %d", lbID)
}
