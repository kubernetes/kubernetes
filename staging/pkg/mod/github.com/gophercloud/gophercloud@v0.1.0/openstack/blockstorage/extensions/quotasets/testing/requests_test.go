package testing

import (
	"errors"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/blockstorage/extensions/quotasets"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	uriQueryParms := map[string]string{}
	HandleSuccessfulRequest(t, "GET", "/os-quota-sets/"+FirstTenantID, getExpectedJSONBody, uriQueryParms)
	actual, err := quotasets.Get(client.ServiceClient(), FirstTenantID).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, &getExpectedQuotaSet, actual)
}

func TestGetUsage(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	uriQueryParms := map[string]string{"usage": "true"}
	HandleSuccessfulRequest(t, "GET", "/os-quota-sets/"+FirstTenantID, getUsageExpectedJSONBody, uriQueryParms)
	actual, err := quotasets.GetUsage(client.ServiceClient(), FirstTenantID).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, getUsageExpectedQuotaSet, actual)
}

func TestFullUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	uriQueryParms := map[string]string{}
	HandleSuccessfulRequest(t, "PUT", "/os-quota-sets/"+FirstTenantID, fullUpdateExpectedJSONBody, uriQueryParms)
	actual, err := quotasets.Update(client.ServiceClient(), FirstTenantID, fullUpdateOpts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, &fullUpdateExpectedQuotaSet, actual)
}

func TestPartialUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	uriQueryParms := map[string]string{}
	HandleSuccessfulRequest(t, "PUT", "/os-quota-sets/"+FirstTenantID, partialUpdateExpectedJSONBody, uriQueryParms)
	actual, err := quotasets.Update(client.ServiceClient(), FirstTenantID, partialUpdateOpts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, &partiualUpdateExpectedQuotaSet, actual)
}

type ErrorUpdateOpts quotasets.UpdateOpts

func (opts ErrorUpdateOpts) ToBlockStorageQuotaUpdateMap() (map[string]interface{}, error) {
	return nil, errors.New("This is an error")
}

func TestErrorInToBlockStorageQuotaUpdateMap(t *testing.T) {
	opts := &ErrorUpdateOpts{}
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleSuccessfulRequest(t, "PUT", "/os-quota-sets/"+FirstTenantID, "", nil)
	_, err := quotasets.Update(client.ServiceClient(), FirstTenantID, opts).Extract()
	if err == nil {
		t.Fatal("Error handling failed")
	}
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDeleteSuccessfully(t)

	err := quotasets.Delete(client.ServiceClient(), FirstTenantID).ExtractErr()
	th.AssertNoErr(t, err)
}
