package throttle

import (
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

const lbID = 12345

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockCreateResponse(t, lbID)

	opts := CreateOpts{MaxConnections: 200}
	err := Create(client.ServiceClient(), lbID, opts).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockGetResponse(t, lbID)

	sp, err := Get(client.ServiceClient(), lbID).Extract()
	th.AssertNoErr(t, err)

	expected := &ConnectionThrottle{MaxConnections: 100}
	th.AssertDeepEquals(t, expected, sp)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockDeleteResponse(t, lbID)

	err := Delete(client.ServiceClient(), lbID).ExtractErr()
	th.AssertNoErr(t, err)
}
