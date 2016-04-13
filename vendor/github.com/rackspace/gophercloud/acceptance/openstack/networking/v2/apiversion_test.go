// +build acceptance networking

package v2

import (
	"testing"

	"github.com/rackspace/gophercloud/openstack/networking/v2/apiversions"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestListAPIVersions(t *testing.T) {
	Setup(t)
	defer Teardown()

	pager := apiversions.ListVersions(Client)
	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		t.Logf("--- Page ---")

		versions, err := apiversions.ExtractAPIVersions(page)
		th.AssertNoErr(t, err)

		for _, v := range versions {
			t.Logf("API Version: ID [%s] Status [%s]", v.ID, v.Status)
		}

		return true, nil
	})
	th.CheckNoErr(t, err)
}

func TestListAPIResources(t *testing.T) {
	Setup(t)
	defer Teardown()

	pager := apiversions.ListVersionResources(Client, "v2.0")
	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		t.Logf("--- Page ---")

		vrs, err := apiversions.ExtractVersionResources(page)
		th.AssertNoErr(t, err)

		for _, vr := range vrs {
			t.Logf("Network: Name [%s] Collection [%s]", vr.Name, vr.Collection)
		}

		return true, nil
	})
	th.CheckNoErr(t, err)
}
