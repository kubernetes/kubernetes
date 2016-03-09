package stacktemplates

import (
	"testing"

	os "github.com/rackspace/gophercloud/openstack/orchestration/v1/stacktemplates"
	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func TestGetTemplate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleGetSuccessfully(t, os.GetOutput)

	actual, err := Get(fake.ServiceClient(), "postman_stack", "16ef0584-4458-41eb-87c8-0dc8d5f66c87").Extract()
	th.AssertNoErr(t, err)

	expected := os.GetExpected
	th.AssertDeepEquals(t, expected, string(actual))
}

func TestValidateTemplate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleValidateSuccessfully(t, os.ValidateOutput)

	opts := os.ValidateOpts{
		Template: `{
			"Description": "Simple template to test heat commands",
			"Parameters": {
				"flavor": {
					"Default": "m1.tiny",
					"Type": "String",
					"NoEcho": "false",
					"Description": "",
					"Label": "flavor"
				}
			}
		}`,
	}
	actual, err := Validate(fake.ServiceClient(), opts).Extract()
	th.AssertNoErr(t, err)

	expected := os.ValidateExpected
	th.AssertDeepEquals(t, expected, actual)
}
