package testing

import (
	"testing"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/container/v1/capsules"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	fakeclient "github.com/gophercloud/gophercloud/testhelper/client"
)

func TestGetCapsule_OldTime(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleCapsuleGetOldTimeSuccessfully(t)

	createdAt, _ := time.Parse(gophercloud.RFC3339ZNoT, "2018-01-12 09:37:25+00:00")
	updatedAt, _ := time.Parse(gophercloud.RFC3339ZNoT, "2018-01-12 09:37:26+00:00")
	startedAt, _ := time.Parse(gophercloud.RFC3339ZNoT, "2018-01-12 09:37:26+00:00")

	ExpectedCapsule.CreatedAt = createdAt
	ExpectedCapsule.UpdatedAt = updatedAt
	ExpectedCapsule.Containers[0].CreatedAt = createdAt
	ExpectedCapsule.Containers[0].UpdatedAt = updatedAt
	ExpectedCapsule.Containers[0].StartedAt = startedAt

	actualCapsule, err := capsules.Get(fakeclient.ServiceClient(), ExpectedCapsule.UUID).Extract()
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, &ExpectedCapsule, actualCapsule)
}

func TestGetCapsule_NewTime(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleCapsuleGetNewTimeSuccessfully(t)

	createdAt, _ := time.Parse(gophercloud.RFC3339ZNoTNoZ, "2018-01-12 09:37:25")
	updatedAt, _ := time.Parse(gophercloud.RFC3339ZNoTNoZ, "2018-01-12 09:37:26")
	startedAt, _ := time.Parse(gophercloud.RFC3339ZNoTNoZ, "2018-01-12 09:37:26")

	ExpectedCapsule.CreatedAt = createdAt
	ExpectedCapsule.UpdatedAt = updatedAt
	ExpectedCapsule.Containers[0].CreatedAt = createdAt
	ExpectedCapsule.Containers[0].UpdatedAt = updatedAt
	ExpectedCapsule.Containers[0].StartedAt = startedAt

	actualCapsule, err := capsules.Get(fakeclient.ServiceClient(), ExpectedCapsule.UUID).Extract()
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, &ExpectedCapsule, actualCapsule)
}

func TestCreateCapsule(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCapsuleCreateSuccessfully(t)

	template := new(capsules.Template)
	template.Bin = []byte(ValidJSONTemplate)

	createOpts := capsules.CreateOpts{
		TemplateOpts: template,
	}
	actualCapsule, err := capsules.Create(fakeclient.ServiceClient(), createOpts).Extract()
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, &ExpectedCapsule, actualCapsule)
}

func TestListCapsule(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleCapsuleListSuccessfully(t)

	createdAt, _ := time.Parse(gophercloud.RFC3339ZNoT, "2018-01-12 09:37:25+00:00")
	updatedAt, _ := time.Parse(gophercloud.RFC3339ZNoT, "2018-01-12 09:37:25+01:00")

	ec := ExpectedCapsule

	ec.CreatedAt = createdAt
	ec.UpdatedAt = updatedAt
	ec.Containers = nil

	expected := []capsules.Capsule{ec}

	count := 0
	results := capsules.List(fakeclient.ServiceClient(), nil)
	err := results.EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := capsules.ExtractCapsules(page)
		if err != nil {
			t.Errorf("Failed to extract capsules: %v", err)
			return false, err
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestListCapsuleV132(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleCapsuleV132ListSuccessfully(t)

	createdAt, _ := time.Parse(gophercloud.RFC3339ZNoTNoZ, "2018-01-12 09:37:25")
	updatedAt, _ := time.Parse(gophercloud.RFC3339ZNoTNoZ, "2018-01-12 09:37:25")

	ec := ExpectedCapsuleV132

	ec.CreatedAt = createdAt
	ec.UpdatedAt = updatedAt
	ec.Containers = nil

	expected := []capsules.CapsuleV132{ec}

	count := 0
	results := capsules.List(fakeclient.ServiceClient(), nil)
	err := results.EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := capsules.ExtractCapsules(page)
		if err != nil {
			t.Errorf("Failed to extract capsules: %v", err)
			return false, err
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleCapsuleDeleteSuccessfully(t)

	res := capsules.Delete(fakeclient.ServiceClient(), "963a239d-3946-452b-be5a-055eab65a421")
	th.AssertNoErr(t, res.Err)
}
