package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/clustering/v1/policies"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func TestListPolicies(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandlePolicyList(t)

	listOpts := policies.ListOpts{
		Limit: 1,
	}

	count := 0
	err := policies.List(fake.ServiceClient(), listOpts).EachPage(func(page pagination.Page) (bool, error) {
		actual, err := policies.ExtractPolicies(page)
		if err != nil {
			t.Errorf("Failed to extract policies: %v", err)
			return false, err
		}

		th.AssertDeepEquals(t, ExpectedPolicies[count], actual)
		count++

		return true, nil
	})

	th.AssertNoErr(t, err)

	if count != 2 {
		t.Errorf("Expected 2 pages, got %d", count)
	}
}

func TestCreatePolicy(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandlePolicyCreate(t)

	expected := ExpectedCreatePolicy

	opts := policies.CreateOpts{
		Name: ExpectedCreatePolicy.Name,
		Spec: ExpectedCreatePolicy.Spec,
	}

	actual, err := policies.Create(fake.ServiceClient(), opts).Extract()
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, &expected, actual)
}

func TestDeletePolicy(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandlePolicyDelete(t)

	res := policies.Delete(fake.ServiceClient(), PolicyIDtoDelete)
	th.AssertNoErr(t, res.ExtractErr())

	requestID := res.Header["X-Openstack-Request-Id"][0]
	th.AssertEquals(t, PolicyDeleteRequestID, requestID)
}

func TestUpdatePolicy(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandlePolicyUpdate(t)

	expected := ExpectedUpdatePolicy

	opts := policies.UpdateOpts{
		Name: ExpectedUpdatePolicy.Name,
	}

	actual, err := policies.Update(fake.ServiceClient(), PolicyIDtoUpdate, opts).Extract()
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, &expected, actual)
}

func TestBadUpdatePolicy(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleBadPolicyUpdate(t)

	opts := policies.UpdateOpts{
		Name: ExpectedUpdatePolicy.Name,
	}

	_, err := policies.Update(fake.ServiceClient(), PolicyIDtoUpdate, opts).Extract()
	th.AssertEquals(t, false, err == nil)
}

func TestValidatePolicy(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandlePolicyValidate(t)

	expected := ExpectedValidatePolicy

	opts := policies.ValidateOpts{
		Spec: ExpectedValidatePolicy.Spec,
	}

	actual, err := policies.Validate(fake.ServiceClient(), opts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, &expected, actual)
}

func TestBadValidatePolicy(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleBadPolicyValidate(t)

	opts := policies.ValidateOpts{
		Spec: ExpectedValidatePolicy.Spec,
	}

	_, err := policies.Validate(fake.ServiceClient(), opts).Extract()
	th.AssertEquals(t, false, err == nil)
}

func TestGetPolicy(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandlePolicyGet(t)

	actual, err := policies.Get(fake.ServiceClient(), PolicyIDtoGet).Extract()
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, ExpectedGetPolicy, *actual)
}
