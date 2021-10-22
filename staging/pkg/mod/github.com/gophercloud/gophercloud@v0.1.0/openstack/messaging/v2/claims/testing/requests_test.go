package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/messaging/v2/claims"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateSuccessfully(t)

	createOpts := claims.CreateOpts{
		TTL:   3600,
		Grace: 3600,
		Limit: 10,
	}

	actual, err := claims.Create(fake.ServiceClient(), QueueName, createOpts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, CreatedClaim, actual)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetSuccessfully(t)

	actual, err := claims.Get(fake.ServiceClient(), QueueName, ClaimID).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, &FirstClaim, actual)
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleUpdateSuccessfully(t)

	updateOpts := claims.UpdateOpts{
		Grace: 1600,
		TTL:   1200,
	}

	err := claims.Update(fake.ServiceClient(), QueueName, ClaimID, updateOpts).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDeleteSuccessfully(t)

	err := claims.Delete(fake.ServiceClient(), QueueName, ClaimID).ExtractErr()
	th.AssertNoErr(t, err)
}
