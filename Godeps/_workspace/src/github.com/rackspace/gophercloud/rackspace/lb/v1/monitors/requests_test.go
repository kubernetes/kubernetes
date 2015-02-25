package monitors

import (
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

const lbID = 12345

func TestUpdateCONNECT(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockUpdateConnectResponse(t, lbID)

	opts := UpdateConnectMonitorOpts{
		AttemptLimit: 3,
		Delay:        10,
		Timeout:      10,
	}

	err := Update(client.ServiceClient(), lbID, opts).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestUpdateHTTP(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockUpdateHTTPResponse(t, lbID)

	opts := UpdateHTTPMonitorOpts{
		AttemptLimit: 3,
		Delay:        10,
		Timeout:      10,
		BodyRegex:    "{regex}",
		Path:         "/foo",
		StatusRegex:  "200",
		Type:         HTTPS,
	}

	err := Update(client.ServiceClient(), lbID, opts).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockGetResponse(t, lbID)

	m, err := Get(client.ServiceClient(), lbID).Extract()
	th.AssertNoErr(t, err)

	expected := &Monitor{
		Type:         CONNECT,
		Delay:        10,
		Timeout:      10,
		AttemptLimit: 3,
	}

	th.AssertDeepEquals(t, expected, m)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockDeleteResponse(t, lbID)

	err := Delete(client.ServiceClient(), lbID).ExtractErr()
	th.AssertNoErr(t, err)
}
