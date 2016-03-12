package defsecrules

import (
	"testing"

	"github.com/rackspace/gophercloud/openstack/compute/v2/extensions/secgroups"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

const ruleID = "{ruleID}"

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockListRulesResponse(t)

	count := 0

	err := List(client.ServiceClient()).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractDefaultRules(page)
		th.AssertNoErr(t, err)

		expected := []DefaultRule{
			DefaultRule{
				FromPort:   80,
				ID:         ruleID,
				IPProtocol: "TCP",
				IPRange:    secgroups.IPRange{CIDR: "10.10.10.0/24"},
				ToPort:     80,
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

	mockCreateRuleResponse(t)

	opts := CreateOpts{
		IPProtocol: "TCP",
		FromPort:   80,
		ToPort:     80,
		CIDR:       "10.10.12.0/24",
	}

	group, err := Create(client.ServiceClient(), opts).Extract()
	th.AssertNoErr(t, err)

	expected := &DefaultRule{
		ID:         ruleID,
		FromPort:   80,
		ToPort:     80,
		IPProtocol: "TCP",
		IPRange:    secgroups.IPRange{CIDR: "10.10.12.0/24"},
	}
	th.AssertDeepEquals(t, expected, group)
}

func TestCreateICMPZero(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockCreateRuleResponseICMPZero(t)

	opts := CreateOpts{
		IPProtocol: "ICMP",
		FromPort:   0,
		ToPort:     0,
		CIDR:       "10.10.12.0/24",
	}

	group, err := Create(client.ServiceClient(), opts).Extract()
	th.AssertNoErr(t, err)

	expected := &DefaultRule{
		ID:         ruleID,
		FromPort:   0,
		ToPort:     0,
		IPProtocol: "ICMP",
		IPRange:    secgroups.IPRange{CIDR: "10.10.12.0/24"},
	}
	th.AssertDeepEquals(t, expected, group)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockGetRuleResponse(t, ruleID)

	group, err := Get(client.ServiceClient(), ruleID).Extract()
	th.AssertNoErr(t, err)

	expected := &DefaultRule{
		ID:         ruleID,
		FromPort:   80,
		ToPort:     80,
		IPProtocol: "TCP",
		IPRange:    secgroups.IPRange{CIDR: "10.10.12.0/24"},
	}

	th.AssertDeepEquals(t, expected, group)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockDeleteRuleResponse(t, ruleID)

	err := Delete(client.ServiceClient(), ruleID).ExtractErr()
	th.AssertNoErr(t, err)
}
