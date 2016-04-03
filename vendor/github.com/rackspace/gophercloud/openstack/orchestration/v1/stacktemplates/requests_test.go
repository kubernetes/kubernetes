package stacktemplates

import (
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func TestGetTemplate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetSuccessfully(t, GetOutput)

	actual, err := Get(fake.ServiceClient(), "postman_stack", "16ef0584-4458-41eb-87c8-0dc8d5f66c87").Extract()
	th.AssertNoErr(t, err)

	expected := GetExpected
	th.AssertDeepEquals(t, expected, string(actual))
}

func TestValidateTemplate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleValidateSuccessfully(t, ValidateOutput)

	opts := ValidateOpts{
		Template: `{
		  "heat_template_version": "2013-05-23",
		  "description": "Simple template to test heat commands",
		  "parameters": {
		    "flavor": {
		      "default": "m1.tiny",
		      "type": "string"
		    }
		  },
		  "resources": {
		    "hello_world": {
		      "type": "OS::Nova::Server",
		      "properties": {
		        "key_name": "heat_key",
		        "flavor": {
		          "get_param": "flavor"
		        },
		        "image": "ad091b52-742f-469e-8f3c-fd81cadf0743",
		        "user_data": "#!/bin/bash -xv\necho \"hello world\" &gt; /root/hello-world.txt\n"
		      }
		    }
		  }
		}`,
	}
	actual, err := Validate(fake.ServiceClient(), opts).Extract()
	th.AssertNoErr(t, err)

	expected := ValidateExpected
	th.AssertDeepEquals(t, expected, actual)
}
