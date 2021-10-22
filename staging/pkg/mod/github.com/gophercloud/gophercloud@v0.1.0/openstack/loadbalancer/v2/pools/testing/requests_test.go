package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/loadbalancer/v2/pools"
	fake "github.com/gophercloud/gophercloud/openstack/loadbalancer/v2/testhelper"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestListPools(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandlePoolListSuccessfully(t)

	pages := 0
	err := pools.List(fake.ServiceClient(), pools.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := pools.ExtractPools(page)
		if err != nil {
			return false, err
		}

		if len(actual) != 2 {
			t.Fatalf("Expected 2 pools, got %d", len(actual))
		}
		th.CheckDeepEquals(t, PoolWeb, actual[0])
		th.CheckDeepEquals(t, PoolDb, actual[1])

		return true, nil
	})

	th.AssertNoErr(t, err)

	if pages != 1 {
		t.Errorf("Expected 1 page, saw %d", pages)
	}
}

func TestListAllPools(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandlePoolListSuccessfully(t)

	allPages, err := pools.List(fake.ServiceClient(), pools.ListOpts{}).AllPages()
	th.AssertNoErr(t, err)
	actual, err := pools.ExtractPools(allPages)
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, PoolWeb, actual[0])
	th.CheckDeepEquals(t, PoolDb, actual[1])
}

func TestCreatePool(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandlePoolCreationSuccessfully(t, SinglePoolBody)

	actual, err := pools.Create(fake.ServiceClient(), pools.CreateOpts{
		LBMethod:       pools.LBMethodRoundRobin,
		Protocol:       "HTTP",
		Name:           "Example pool",
		ProjectID:      "2ffc6e22aae24e4795f87155d24c896f",
		LoadbalancerID: "79e05663-7f03-45d2-a092-8b94062f22ab",
	}).Extract()
	th.AssertNoErr(t, err)

	th.CheckDeepEquals(t, PoolDb, *actual)
}

func TestGetPool(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandlePoolGetSuccessfully(t)

	client := fake.ServiceClient()
	actual, err := pools.Get(client, "c3741b06-df4d-4715-b142-276b6bce75ab").Extract()
	if err != nil {
		t.Fatalf("Unexpected Get error: %v", err)
	}

	th.CheckDeepEquals(t, PoolDb, *actual)
}

func TestDeletePool(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandlePoolDeletionSuccessfully(t)

	res := pools.Delete(fake.ServiceClient(), "c3741b06-df4d-4715-b142-276b6bce75ab")
	th.AssertNoErr(t, res.Err)
}

func TestUpdatePool(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandlePoolUpdateSuccessfully(t)

	client := fake.ServiceClient()
	name := "NewPoolName"
	actual, err := pools.Update(client, "c3741b06-df4d-4715-b142-276b6bce75ab", pools.UpdateOpts{
		Name:     &name,
		LBMethod: pools.LBMethodLeastConnections,
	}).Extract()
	if err != nil {
		t.Fatalf("Unexpected Update error: %v", err)
	}

	th.CheckDeepEquals(t, PoolUpdated, *actual)
}

func TestRequiredPoolCreateOpts(t *testing.T) {
	res := pools.Create(fake.ServiceClient(), pools.CreateOpts{})
	if res.Err == nil {
		t.Fatalf("Expected error, got none")
	}
	res = pools.Create(fake.ServiceClient(), pools.CreateOpts{
		LBMethod:       pools.LBMethod("invalid"),
		Protocol:       pools.ProtocolHTTPS,
		LoadbalancerID: "69055154-f603-4a28-8951-7cc2d9e54a9a",
	})
	if res.Err == nil {
		t.Fatalf("Expected error, but got none")
	}

	res = pools.Create(fake.ServiceClient(), pools.CreateOpts{
		LBMethod:       pools.LBMethodRoundRobin,
		Protocol:       pools.Protocol("invalid"),
		LoadbalancerID: "69055154-f603-4a28-8951-7cc2d9e54a9a",
	})
	if res.Err == nil {
		t.Fatalf("Expected error, but got none")
	}

	res = pools.Create(fake.ServiceClient(), pools.CreateOpts{
		LBMethod: pools.LBMethodRoundRobin,
		Protocol: pools.ProtocolHTTPS,
	})
	if res.Err == nil {
		t.Fatalf("Expected error, but got none")
	}
}

func TestListMembers(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleMemberListSuccessfully(t)

	pages := 0
	err := pools.ListMembers(fake.ServiceClient(), "332abe93-f488-41ba-870b-2ac66be7f853", pools.ListMembersOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := pools.ExtractMembers(page)
		if err != nil {
			return false, err
		}

		if len(actual) != 2 {
			t.Fatalf("Expected 2 members, got %d", len(actual))
		}
		th.CheckDeepEquals(t, MemberWeb, actual[0])
		th.CheckDeepEquals(t, MemberDb, actual[1])

		return true, nil
	})

	th.AssertNoErr(t, err)

	if pages != 1 {
		t.Errorf("Expected 1 page, saw %d", pages)
	}
}

func TestListAllMembers(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleMemberListSuccessfully(t)

	allPages, err := pools.ListMembers(fake.ServiceClient(), "332abe93-f488-41ba-870b-2ac66be7f853", pools.ListMembersOpts{}).AllPages()
	th.AssertNoErr(t, err)
	actual, err := pools.ExtractMembers(allPages)
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, MemberWeb, actual[0])
	th.CheckDeepEquals(t, MemberDb, actual[1])
}

func TestCreateMember(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleMemberCreationSuccessfully(t, SingleMemberBody)

	weight := 10
	actual, err := pools.CreateMember(fake.ServiceClient(), "332abe93-f488-41ba-870b-2ac66be7f853", pools.CreateMemberOpts{
		Name:         "db",
		SubnetID:     "1981f108-3c48-48d2-b908-30f7d28532c9",
		ProjectID:    "2ffc6e22aae24e4795f87155d24c896f",
		Address:      "10.0.2.11",
		ProtocolPort: 80,
		Weight:       &weight,
	}).Extract()
	th.AssertNoErr(t, err)

	th.CheckDeepEquals(t, MemberDb, *actual)
}

func TestRequiredMemberCreateOpts(t *testing.T) {
	res := pools.CreateMember(fake.ServiceClient(), "", pools.CreateMemberOpts{})
	if res.Err == nil {
		t.Fatalf("Expected error, got none")
	}
	res = pools.CreateMember(fake.ServiceClient(), "", pools.CreateMemberOpts{Address: "1.2.3.4", ProtocolPort: 80})
	if res.Err == nil {
		t.Fatalf("Expected error, but got none")
	}
	res = pools.CreateMember(fake.ServiceClient(), "332abe93-f488-41ba-870b-2ac66be7f853", pools.CreateMemberOpts{ProtocolPort: 80})
	if res.Err == nil {
		t.Fatalf("Expected error, but got none")
	}
	res = pools.CreateMember(fake.ServiceClient(), "332abe93-f488-41ba-870b-2ac66be7f853", pools.CreateMemberOpts{Address: "1.2.3.4"})
	if res.Err == nil {
		t.Fatalf("Expected error, but got none")
	}
}

func TestGetMember(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleMemberGetSuccessfully(t)

	client := fake.ServiceClient()
	actual, err := pools.GetMember(client, "332abe93-f488-41ba-870b-2ac66be7f853", "2a280670-c202-4b0b-a562-34077415aabf").Extract()
	if err != nil {
		t.Fatalf("Unexpected Get error: %v", err)
	}

	th.CheckDeepEquals(t, MemberDb, *actual)
}

func TestDeleteMember(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleMemberDeletionSuccessfully(t)

	res := pools.DeleteMember(fake.ServiceClient(), "332abe93-f488-41ba-870b-2ac66be7f853", "2a280670-c202-4b0b-a562-34077415aabf")
	th.AssertNoErr(t, res.Err)
}

func TestUpdateMember(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleMemberUpdateSuccessfully(t)

	weight := 4
	client := fake.ServiceClient()
	name := "newMemberName"
	actual, err := pools.UpdateMember(client, "332abe93-f488-41ba-870b-2ac66be7f853", "2a280670-c202-4b0b-a562-34077415aabf", pools.UpdateMemberOpts{
		Name:   &name,
		Weight: &weight,
	}).Extract()
	if err != nil {
		t.Fatalf("Unexpected Update error: %v", err)
	}

	th.CheckDeepEquals(t, MemberUpdated, *actual)
}

func TestBatchUpdateMembers(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleMembersUpdateSuccessfully(t)

	weight_1 := 20
	member1 := pools.BatchUpdateMemberOpts{
		Address:      "192.0.2.16",
		ProtocolPort: 80,
		Name:         "web-server-1",
		SubnetID:     "bbb35f84-35cc-4b2f-84c2-a6a29bba68aa",
		Weight:       &weight_1,
	}

	weight_2 := 10
	member2 := pools.BatchUpdateMemberOpts{
		Address:      "192.0.2.17",
		ProtocolPort: 80,
		Name:         "web-server-2",
		Weight:       &weight_2,
		SubnetID:     "bbb35f84-35cc-4b2f-84c2-a6a29bba68aa",
	}
	members := []pools.BatchUpdateMemberOpts{member1, member2}

	res := pools.BatchUpdateMembers(fake.ServiceClient(), "332abe93-f488-41ba-870b-2ac66be7f853", members)
	th.AssertNoErr(t, res.Err)
}

func TestRequiredBatchUpdateMemberOpts(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	res := pools.BatchUpdateMembers(fake.ServiceClient(), "332abe93-f488-41ba-870b-2ac66be7f853", []pools.BatchUpdateMemberOpts{
		{
			Name: "web-server-1",
		},
	})
	if res.Err == nil {
		t.Fatalf("Expected error, but got none")
	}

	res = pools.BatchUpdateMembers(fake.ServiceClient(), "332abe93-f488-41ba-870b-2ac66be7f853", []pools.BatchUpdateMemberOpts{
		{
			Address: "192.0.2.17",
			Name:    "web-server-1",
		},
	})
	if res.Err == nil {
		t.Fatalf("Expected error, but got none")
	}

	res = pools.BatchUpdateMembers(fake.ServiceClient(), "332abe93-f488-41ba-870b-2ac66be7f853", []pools.BatchUpdateMemberOpts{
		{
			ProtocolPort: 80,
			Name:         "web-server-1",
		},
	})
	if res.Err == nil {
		t.Fatalf("Expected error, but got none")
	}
}
