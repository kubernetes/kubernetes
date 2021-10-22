package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/secgroups"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	"github.com/gophercloud/gophercloud/testhelper/client"
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

	err := secgroups.List(client.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := secgroups.ExtractSecurityGroups(page)
		if err != nil {
			t.Errorf("Failed to extract users: %v", err)
			return false, err
		}

		expected := []secgroups.SecurityGroup{
			{
				ID:          groupID,
				Description: "default",
				Name:        "default",
				Rules:       []secgroups.Rule{},
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

	err := secgroups.ListByServer(client.ServiceClient(), serverID).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := secgroups.ExtractSecurityGroups(page)
		if err != nil {
			t.Errorf("Failed to extract users: %v", err)
			return false, err
		}

		expected := []secgroups.SecurityGroup{
			{
				ID:          groupID,
				Description: "default",
				Name:        "default",
				Rules:       []secgroups.Rule{},
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

	opts := secgroups.CreateOpts{
		Name:        "test",
		Description: "something",
	}

	group, err := secgroups.Create(client.ServiceClient(), opts).Extract()
	th.AssertNoErr(t, err)

	expected := &secgroups.SecurityGroup{
		ID:          groupID,
		Name:        "test",
		Description: "something",
		TenantID:    "openstack",
		Rules:       []secgroups.Rule{},
	}
	th.AssertDeepEquals(t, expected, group)
}

func TestUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockUpdateGroupResponse(t, groupID)

	description := "new_desc"
	opts := secgroups.UpdateOpts{
		Name:        "new_name",
		Description: &description,
	}

	group, err := secgroups.Update(client.ServiceClient(), groupID, opts).Extract()
	th.AssertNoErr(t, err)

	expected := &secgroups.SecurityGroup{
		ID:          groupID,
		Name:        "new_name",
		Description: "something",
		TenantID:    "openstack",
		Rules:       []secgroups.Rule{},
	}
	th.AssertDeepEquals(t, expected, group)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockGetGroupsResponse(t, groupID)

	group, err := secgroups.Get(client.ServiceClient(), groupID).Extract()
	th.AssertNoErr(t, err)

	expected := &secgroups.SecurityGroup{
		ID:          groupID,
		Description: "default",
		Name:        "default",
		TenantID:    "openstack",
		Rules: []secgroups.Rule{
			{
				FromPort:      80,
				ToPort:        85,
				IPProtocol:    "TCP",
				IPRange:       secgroups.IPRange{CIDR: "0.0.0.0"},
				Group:         secgroups.Group{TenantID: "openstack", Name: "default"},
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

	group, err := secgroups.Get(client.ServiceClient(), "12345").Extract()
	th.AssertNoErr(t, err)

	expected := &secgroups.SecurityGroup{ID: "12345"}
	th.AssertDeepEquals(t, expected, group)
}

func TestGetNumericRuleID(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	numericGroupID := 12345

	mockGetNumericIDGroupRuleResponse(t, numericGroupID)

	group, err := secgroups.Get(client.ServiceClient(), "12345").Extract()
	th.AssertNoErr(t, err)

	expected := &secgroups.SecurityGroup{
		ID: "12345",
		Rules: []secgroups.Rule{
			{
				ParentGroupID: "12345",
				ID:            "12345",
			},
		},
	}
	th.AssertDeepEquals(t, expected, group)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockDeleteGroupResponse(t, groupID)

	err := secgroups.Delete(client.ServiceClient(), groupID).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestAddRule(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockAddRuleResponse(t)

	opts := secgroups.CreateRuleOpts{
		ParentGroupID: groupID,
		FromPort:      22,
		ToPort:        22,
		IPProtocol:    "TCP",
		CIDR:          "0.0.0.0/0",
	}

	rule, err := secgroups.CreateRule(client.ServiceClient(), opts).Extract()
	th.AssertNoErr(t, err)

	expected := &secgroups.Rule{
		FromPort:      22,
		ToPort:        22,
		Group:         secgroups.Group{},
		IPProtocol:    "TCP",
		ParentGroupID: groupID,
		IPRange:       secgroups.IPRange{CIDR: "0.0.0.0/0"},
		ID:            ruleID,
	}

	th.AssertDeepEquals(t, expected, rule)
}

func TestAddRuleICMPZero(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockAddRuleResponseICMPZero(t)

	opts := secgroups.CreateRuleOpts{
		ParentGroupID: groupID,
		FromPort:      0,
		ToPort:        0,
		IPProtocol:    "ICMP",
		CIDR:          "0.0.0.0/0",
	}

	rule, err := secgroups.CreateRule(client.ServiceClient(), opts).Extract()
	th.AssertNoErr(t, err)

	expected := &secgroups.Rule{
		FromPort:      0,
		ToPort:        0,
		Group:         secgroups.Group{},
		IPProtocol:    "ICMP",
		ParentGroupID: groupID,
		IPRange:       secgroups.IPRange{CIDR: "0.0.0.0/0"},
		ID:            ruleID,
	}

	th.AssertDeepEquals(t, expected, rule)
}

func TestDeleteRule(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockDeleteRuleResponse(t, ruleID)

	err := secgroups.DeleteRule(client.ServiceClient(), ruleID).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestAddServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockAddServerToGroupResponse(t, serverID)

	err := secgroups.AddServer(client.ServiceClient(), serverID, "test").ExtractErr()
	th.AssertNoErr(t, err)
}

func TestRemoveServer(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockRemoveServerFromGroupResponse(t, serverID)

	err := secgroups.RemoveServer(client.ServiceClient(), serverID, "test").ExtractErr()
	th.AssertNoErr(t, err)
}
