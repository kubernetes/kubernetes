package secgroups

import (
	"testing"

	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

const (
	serverID = "{serverID}"
	groupID  = "{groupID}"
	ruleID   = "{ruleID}"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockListGroupsResponse(t)

	count := 0

	err := List(client.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractSecurityGroups(page)
		if err != nil {
			t.Errorf("Failed to extract users: %v", err)
			return false, err
		}

		expected := []SecurityGroup{
			SecurityGroup{
				ID:          groupID,
				Description: "default",
				Name:        "default",
				Rules:       []Rule{},
				TenantID:    "openstack",
			},
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, count)
}

func TestListByServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockListGroupsByServerResponse(t, serverID)

	count := 0

	err := ListByServer(client.ServiceClient(), serverID).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractSecurityGroups(page)
		if err != nil {
			t.Errorf("Failed to extract users: %v", err)
			return false, err
		}

		expected := []SecurityGroup{
			SecurityGroup{
				ID:          groupID,
				Description: "default",
				Name:        "default",
				Rules:       []Rule{},
				TenantID:    "openstack",
			},
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, count)
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockCreateGroupResponse(t)

	opts := CreateOpts{
		Name:        "test",
		Description: "something",
	}

	group, err := Create(client.ServiceClient(), opts).Extract()
	th.AssertNoErr(t, err)

	expected := &SecurityGroup{
		ID:          groupID,
		Name:        "test",
		Description: "something",
		TenantID:    "openstack",
		Rules:       []Rule{},
	}
	th.AssertDeepEquals(t, expected, group)
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockUpdateGroupResponse(t, groupID)

	opts := UpdateOpts{
		Name:        "new_name",
		Description: "new_desc",
	}

	group, err := Update(client.ServiceClient(), groupID, opts).Extract()
	th.AssertNoErr(t, err)

	expected := &SecurityGroup{
		ID:          groupID,
		Name:        "new_name",
		Description: "something",
		TenantID:    "openstack",
		Rules:       []Rule{},
	}
	th.AssertDeepEquals(t, expected, group)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockGetGroupsResponse(t, groupID)

	group, err := Get(client.ServiceClient(), groupID).Extract()
	th.AssertNoErr(t, err)

	expected := &SecurityGroup{
		ID:          groupID,
		Description: "default",
		Name:        "default",
		TenantID:    "openstack",
		Rules: []Rule{
			Rule{
				FromPort:      80,
				ToPort:        85,
				IPProtocol:    "TCP",
				IPRange:       IPRange{CIDR: "0.0.0.0"},
				Group:         Group{TenantID: "openstack", Name: "default"},
				ParentGroupID: groupID,
				ID:            ruleID,
			},
		},
	}

	th.AssertDeepEquals(t, expected, group)
}

func TestGetNumericID(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	numericGroupID := 12345

	mockGetNumericIDGroupResponse(t, numericGroupID)

	group, err := Get(client.ServiceClient(), "12345").Extract()
	th.AssertNoErr(t, err)

	expected := &SecurityGroup{ID: "12345"}
	th.AssertDeepEquals(t, expected, group)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockDeleteGroupResponse(t, groupID)

	err := Delete(client.ServiceClient(), groupID).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestAddRule(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockAddRuleResponse(t)

	opts := CreateRuleOpts{
		ParentGroupID: groupID,
		FromPort:      22,
		ToPort:        22,
		IPProtocol:    "TCP",
		CIDR:          "0.0.0.0/0",
	}

	rule, err := CreateRule(client.ServiceClient(), opts).Extract()
	th.AssertNoErr(t, err)

	expected := &Rule{
		FromPort:      22,
		ToPort:        22,
		Group:         Group{},
		IPProtocol:    "TCP",
		ParentGroupID: groupID,
		IPRange:       IPRange{CIDR: "0.0.0.0/0"},
		ID:            ruleID,
	}

	th.AssertDeepEquals(t, expected, rule)
}

func TestAddRuleICMPZero(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockAddRuleResponseICMPZero(t)

	opts := CreateRuleOpts{
		ParentGroupID: groupID,
		FromPort:      0,
		ToPort:        0,
		IPProtocol:    "ICMP",
		CIDR:          "0.0.0.0/0",
	}

	rule, err := CreateRule(client.ServiceClient(), opts).Extract()
	th.AssertNoErr(t, err)

	expected := &Rule{
		FromPort:      0,
		ToPort:        0,
		Group:         Group{},
		IPProtocol:    "ICMP",
		ParentGroupID: groupID,
		IPRange:       IPRange{CIDR: "0.0.0.0/0"},
		ID:            ruleID,
	}

	th.AssertDeepEquals(t, expected, rule)
}

func TestDeleteRule(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockDeleteRuleResponse(t, ruleID)

	err := DeleteRule(client.ServiceClient(), ruleID).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestAddServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockAddServerToGroupResponse(t, serverID)

	err := AddServerToGroup(client.ServiceClient(), serverID, "test").ExtractErr()
	th.AssertNoErr(t, err)
}

func TestRemoveServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockRemoveServerFromGroupResponse(t, serverID)

	err := RemoveServerFromGroup(client.ServiceClient(), serverID, "test").ExtractErr()
	th.AssertNoErr(t, err)
}
