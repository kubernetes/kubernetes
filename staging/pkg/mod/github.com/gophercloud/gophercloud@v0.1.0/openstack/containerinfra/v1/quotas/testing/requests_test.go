package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/containerinfra/v1/quotas"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func TestCreateQuota(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleCreateQuotaSuccessfully(t)

	opts := quotas.CreateOpts{
		ProjectID: "aa5436ab58144c768ca4e9d2e9f5c3b2",
		Resource:  "Cluster",
		HardLimit: 10,
	}

	sc := fake.ServiceClient()
	sc.Endpoint = sc.Endpoint + "v1/"

	res := quotas.Create(sc, opts)
	th.AssertNoErr(t, res.Err)

	requestID := res.Header.Get("X-OpenStack-Request-Id")
	th.AssertEquals(t, requestUUID, requestID)

	quota, err := res.Extract()
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, projectID, quota.ProjectID)
}
