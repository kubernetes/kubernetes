package testing

import (
	"testing"

	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/l7policies"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestCreateL7Policy(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleL7PolicyCreationSuccessfully(t, SingleL7PolicyBody)

	actual, err := l7policies.Create(fake.ServiceClient(), l7policies.CreateOpts{
		Name:        "redirect-example.com",
		ListenerID:  "023f2e34-7806-443b-bfae-16c324569a3d",
		Action:      l7policies.ActionRedirectToURL,
		RedirectURL: "http://www.example.com",
	}).Extract()

	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, L7PolicyToURL, *actual)
}

func TestRequiredL7PolicyCreateOpts(t *testing.T) {
	// no param specified.
	res := l7policies.Create(fake.ServiceClient(), l7policies.CreateOpts{})
	if res.Err == nil {
		t.Fatalf("Expected error, got none")
	}

	// Action is invalid.
	res = l7policies.Create(fake.ServiceClient(), l7policies.CreateOpts{
		ListenerID: "023f2e34-7806-443b-bfae-16c324569a3d",
		Action:     l7policies.Action("invalid"),
	})
	if res.Err == nil {
		t.Fatalf("Expected error, but got none")
	}
}

func TestListL7Policies(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleL7PolicyListSuccessfully(t)

	pages := 0
	err := l7policies.List(fake.ServiceClient(), l7policies.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := l7policies.ExtractL7Policies(page)
		if err != nil {
			return false, err
		}

		if len(actual) != 2 {
			t.Fatalf("Expected 2 l7policies, got %d", len(actual))
		}
		th.CheckDeepEquals(t, L7PolicyToURL, actual[0])
		th.CheckDeepEquals(t, L7PolicyToPool, actual[1])

		return true, nil
	})

	th.AssertNoErr(t, err)

	if pages != 1 {
		t.Errorf("Expected 1 page, saw %d", pages)
	}
}

func TestListAllL7Policies(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleL7PolicyListSuccessfully(t)

	allPages, err := l7policies.List(fake.ServiceClient(), l7policies.ListOpts{}).AllPages()
	th.AssertNoErr(t, err)
	actual, err := l7policies.ExtractL7Policies(allPages)
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, L7PolicyToURL, actual[0])
	th.CheckDeepEquals(t, L7PolicyToPool, actual[1])
}

func TestGetL7Policy(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleL7PolicyGetSuccessfully(t)

	client := fake.ServiceClient()
	actual, err := l7policies.Get(client, "8a1412f0-4c32-4257-8b07-af4770b604fd").Extract()
	if err != nil {
		t.Fatalf("Unexpected Get error: %v", err)
	}

	th.CheckDeepEquals(t, L7PolicyToURL, *actual)
}

func TestDeleteL7Policy(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleL7PolicyDeletionSuccessfully(t)

	res := l7policies.Delete(fake.ServiceClient(), "8a1412f0-4c32-4257-8b07-af4770b604fd")
	th.AssertNoErr(t, res.Err)
}

func TestUpdateL7Policy(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleL7PolicyUpdateSuccessfully(t)

	client := fake.ServiceClient()
	newName := "NewL7PolicyName"
	redirectURL := "http://www.new-example.com"
	actual, err := l7policies.Update(client, "8a1412f0-4c32-4257-8b07-af4770b604fd",
		l7policies.UpdateOpts{
			Name:        &newName,
			Action:      l7policies.ActionRedirectToURL,
			RedirectURL: &redirectURL,
		}).Extract()
	if err != nil {
		t.Fatalf("Unexpected Update error: %v", err)
	}

	th.CheckDeepEquals(t, L7PolicyUpdated, *actual)
}

func TestUpdateL7PolicyNullRedirectURL(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleL7PolicyUpdateNullRedirectURLSuccessfully(t)

	client := fake.ServiceClient()
	newName := "NewL7PolicyName"
	redirectURL := ""
	actual, err := l7policies.Update(client, "8a1412f0-4c32-4257-8b07-af4770b604fd",
		l7policies.UpdateOpts{
			Name:        &newName,
			RedirectURL: &redirectURL,
		}).Extract()
	if err != nil {
		t.Fatalf("Unexpected Update error: %v", err)
	}

	th.CheckDeepEquals(t, L7PolicyNullRedirectURLUpdated, *actual)
}

func TestUpdateL7PolicyWithInvalidOpts(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	res := l7policies.Update(fake.ServiceClient(), "8a1412f0-4c32-4257-8b07-af4770b604fd", l7policies.UpdateOpts{
		Action: l7policies.Action("invalid"),
	})
	if res.Err == nil {
		t.Fatalf("Expected error, got none")
	}
}

func TestCreateRule(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleRuleCreationSuccessfully(t, SingleRuleBody)

	actual, err := l7policies.CreateRule(fake.ServiceClient(), "8a1412f0-4c32-4257-8b07-af4770b604fd", l7policies.CreateRuleOpts{
		RuleType:    l7policies.TypePath,
		CompareType: l7policies.CompareTypeRegex,
		Value:       "/images*",
	}).Extract()
	th.AssertNoErr(t, err)

	th.CheckDeepEquals(t, RulePath, *actual)
}

func TestRequiredRuleCreateOpts(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	res := l7policies.CreateRule(fake.ServiceClient(), "", l7policies.CreateRuleOpts{})
	if res.Err == nil {
		t.Fatalf("Expected error, got none")
	}
	res = l7policies.CreateRule(fake.ServiceClient(), "8a1412f0-4c32-4257-8b07-af4770b604fd", l7policies.CreateRuleOpts{
		RuleType: l7policies.TypePath,
	})
	if res.Err == nil {
		t.Fatalf("Expected error, but got none")
	}
	res = l7policies.CreateRule(fake.ServiceClient(), "8a1412f0-4c32-4257-8b07-af4770b604fd", l7policies.CreateRuleOpts{
		RuleType:    l7policies.RuleType("invalid"),
		CompareType: l7policies.CompareTypeRegex,
		Value:       "/images*",
	})
	if res.Err == nil {
		t.Fatalf("Expected error, but got none")
	}
	res = l7policies.CreateRule(fake.ServiceClient(), "8a1412f0-4c32-4257-8b07-af4770b604fd", l7policies.CreateRuleOpts{
		RuleType:    l7policies.TypePath,
		CompareType: l7policies.CompareType("invalid"),
		Value:       "/images*",
	})
	if res.Err == nil {
		t.Fatalf("Expected error, but got none")
	}
}

func TestListRules(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleRuleListSuccessfully(t)

	pages := 0
	err := l7policies.ListRules(fake.ServiceClient(), "8a1412f0-4c32-4257-8b07-af4770b604fd", l7policies.ListRulesOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := l7policies.ExtractRules(page)
		if err != nil {
			return false, err
		}

		if len(actual) != 2 {
			t.Fatalf("Expected 2 rules, got %d", len(actual))
		}
		th.CheckDeepEquals(t, RulePath, actual[0])
		th.CheckDeepEquals(t, RuleHostName, actual[1])

		return true, nil
	})

	th.AssertNoErr(t, err)

	if pages != 1 {
		t.Errorf("Expected 1 page, saw %d", pages)
	}
}

func TestListAllRules(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleRuleListSuccessfully(t)

	allPages, err := l7policies.ListRules(fake.ServiceClient(), "8a1412f0-4c32-4257-8b07-af4770b604fd", l7policies.ListRulesOpts{}).AllPages()
	th.AssertNoErr(t, err)

	actual, err := l7policies.ExtractRules(allPages)
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, RulePath, actual[0])
	th.CheckDeepEquals(t, RuleHostName, actual[1])
}

func TestGetRule(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleRuleGetSuccessfully(t)

	client := fake.ServiceClient()
	actual, err := l7policies.GetRule(client, "8a1412f0-4c32-4257-8b07-af4770b604fd", "16621dbb-a736-4888-a57a-3ecd53df784c").Extract()
	if err != nil {
		t.Fatalf("Unexpected Get error: %v", err)
	}

	th.CheckDeepEquals(t, RulePath, *actual)
}

func TestDeleteRule(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleRuleDeletionSuccessfully(t)

	res := l7policies.DeleteRule(fake.ServiceClient(), "8a1412f0-4c32-4257-8b07-af4770b604fd", "16621dbb-a736-4888-a57a-3ecd53df784c")
	th.AssertNoErr(t, res.Err)
}

func TestUpdateRule(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleRuleUpdateSuccessfully(t)

	client := fake.ServiceClient()
	invert := false
	key := ""
	actual, err := l7policies.UpdateRule(client, "8a1412f0-4c32-4257-8b07-af4770b604fd", "16621dbb-a736-4888-a57a-3ecd53df784c", l7policies.UpdateRuleOpts{
		RuleType:    l7policies.TypePath,
		CompareType: l7policies.CompareTypeRegex,
		Value:       "/images/special*",
		Invert:      &invert,
		Key:         &key,
	}).Extract()
	if err != nil {
		t.Fatalf("Unexpected Update error: %v", err)
	}

	th.CheckDeepEquals(t, RuleUpdated, *actual)
}

func TestUpdateRuleWithInvalidOpts(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	res := l7policies.UpdateRule(fake.ServiceClient(), "", "", l7policies.UpdateRuleOpts{
		RuleType: l7policies.RuleType("invalid"),
	})
	if res.Err == nil {
		t.Fatalf("Expected error, got none")
	}

	res = l7policies.UpdateRule(fake.ServiceClient(), "", "", l7policies.UpdateRuleOpts{
		CompareType: l7policies.CompareType("invalid"),
	})
	if res.Err == nil {
		t.Fatalf("Expected error, got none")
	}
}
