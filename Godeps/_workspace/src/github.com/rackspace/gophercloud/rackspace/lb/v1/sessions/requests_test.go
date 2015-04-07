package sessions

import (
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

const lbID = 12345

func TestEnable(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockEnableResponse(t, lbID)

	opts := CreateOpts{Type: HTTPCOOKIE}
	err := Enable(client.ServiceClient(), lbID, opts).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockGetResponse(t, lbID)

	sp, err := Get(client.ServiceClient(), lbID).Extract()
	th.AssertNoErr(t, err)

	expected := &SessionPersistence{Type: HTTPCOOKIE}
	th.AssertDeepEquals(t, expected, sp)
}

func TestDisable(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockDisableResponse(t, lbID)

	err := Disable(client.ServiceClient(), lbID).ExtractErr()
	th.AssertNoErr(t, err)
}
