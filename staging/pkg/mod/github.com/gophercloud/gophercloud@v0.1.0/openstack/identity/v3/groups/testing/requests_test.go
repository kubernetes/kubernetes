package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/identity/v3/groups"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListGroups(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListGroupsSuccessfully(t)

	count := 0
	err := groups.List(client.ServiceClient(), nil).EachPage(func(page pagination.Page) (bool, error) {
		count++

		actual, err := groups.ExtractGroups(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, ExpectedGroupsSlice, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}

func TestListGroupsAllPages(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListGroupsSuccessfully(t)

	allPages, err := groups.List(client.ServiceClient(), nil).AllPages()
	th.AssertNoErr(t, err)
	actual, err := groups.ExtractGroups(allPages)
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, ExpectedGroupsSlice, actual)
	th.AssertEquals(t, ExpectedGroupsSlice[0].Extra["email"], "support@localhost")
	th.AssertEquals(t, ExpectedGroupsSlice[1].Extra["email"], "support@example.com")
}

func TestListGroupsFiltersCheck(t *testing.T) {
	type test struct {
		filterName string
		wantErr    bool
	}
	tests := []test{
		{"foo__contains", false},
		{"foo", true},
		{"foo_contains", true},
		{"foo__", true},
		{"__foo", true},
	}

	var listOpts groups.ListOpts
	for _, _test := range tests {
		listOpts.Filters = map[string]string{_test.filterName: "bar"}
		_, err := listOpts.ToGroupListQuery()

		if !_test.wantErr {
			th.AssertNoErr(t, err)
		} else {
			switch _t := err.(type) {
			case nil:
				t.Fatal("error expected but got a nil")
			case groups.InvalidListFilter:
			default:
				t.Fatalf("unexpected error type: [%T]", _t)
			}
		}
	}
}

func TestGetGroup(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetGroupSuccessfully(t)

	actual, err := groups.Get(client.ServiceClient(), "9fe1d3").Extract()

	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, SecondGroup, *actual)
	th.AssertEquals(t, SecondGroup.Extra["email"], "support@example.com")
}

func TestCreateGroup(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateGroupSuccessfully(t)

	createOpts := groups.CreateOpts{
		Name:        "support",
		DomainID:    "1789d1",
		Description: "group for support users",
		Extra: map[string]interface{}{
			"email": "support@example.com",
		},
	}

	actual, err := groups.Create(client.ServiceClient(), createOpts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, SecondGroup, *actual)
}

func TestUpdateGroup(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleUpdateGroupSuccessfully(t)

	var description = "L2 Support Team"
	updateOpts := groups.UpdateOpts{
		Description: &description,
		Extra: map[string]interface{}{
			"email": "supportteam@example.com",
		},
	}

	actual, err := groups.Update(client.ServiceClient(), "9fe1d3", updateOpts).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, SecondGroupUpdated, *actual)
}

func TestDeleteGroup(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDeleteGroupSuccessfully(t)

	res := groups.Delete(client.ServiceClient(), "9fe1d3")
	th.AssertNoErr(t, res.Err)
}
