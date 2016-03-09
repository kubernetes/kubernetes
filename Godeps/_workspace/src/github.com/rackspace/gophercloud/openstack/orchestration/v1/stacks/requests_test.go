package stacks

import (
	"testing"

	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func TestCreateStack(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateSuccessfully(t, CreateOutput)

	createOpts := CreateOpts{
		Name:    "stackcreated",
		Timeout: 60,
		Template: `
    {
      "stack_name": "postman_stack",
      "template": {
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
            "type":"OS::Nova::Server",
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
      }
    }`,
		DisableRollback: Disable,
	}
	actual, err := Create(fake.ServiceClient(), createOpts).Extract()
	th.AssertNoErr(t, err)

	expected := CreateExpected
	th.AssertDeepEquals(t, expected, actual)
}

func TestCreateStackNewTemplateFormat(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateSuccessfully(t, CreateOutput)
	template := new(Template)
	template.Bin = []byte(`
		{
			"heat_template_version": "2013-05-23",
			"description": "Simple template to test heat commands",
			"parameters": {
				"flavor": {
					"default": "m1.tiny",
					"type": "string"
				}
			}
		}`)
	createOpts := CreateOpts{
		Name:            "stackcreated",
		Timeout:         60,
		TemplateOpts:    template,
		DisableRollback: Disable,
	}
	actual, err := Create(fake.ServiceClient(), createOpts).Extract()
	th.AssertNoErr(t, err)

	expected := CreateExpected
	th.AssertDeepEquals(t, expected, actual)
}

func TestAdoptStack(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateSuccessfully(t, CreateOutput)

	adoptOpts := AdoptOpts{
		AdoptStackData: `{environment{parameters{}}}`,
		Name:           "stackcreated",
		Timeout:        60,
		Template: `
    {
      "stack_name": "postman_stack",
      "template": {
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
            "type":"OS::Nova::Server",
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
      }
    }`,
		DisableRollback: Disable,
	}
	actual, err := Adopt(fake.ServiceClient(), adoptOpts).Extract()
	th.AssertNoErr(t, err)

	expected := CreateExpected
	th.AssertDeepEquals(t, expected, actual)
}

func TestAdoptStackNewTemplateFormat(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleCreateSuccessfully(t, CreateOutput)
	template := new(Template)
	template.Bin = []byte(`
{
  "stack_name": "postman_stack",
  "template": {
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
		"type":"OS::Nova::Server",
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
  }
}`)
	adoptOpts := AdoptOpts{
		AdoptStackData:  `{environment{parameters{}}}`,
		Name:            "stackcreated",
		Timeout:         60,
		TemplateOpts:    template,
		DisableRollback: Disable,
	}
	actual, err := Adopt(fake.ServiceClient(), adoptOpts).Extract()
	th.AssertNoErr(t, err)

	expected := CreateExpected
	th.AssertDeepEquals(t, expected, actual)
}

func TestListStack(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleListSuccessfully(t, FullListOutput)

	count := 0
	err := List(fake.ServiceClient(), nil).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractStacks(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, ListExpected, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}

func TestGetStack(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleGetSuccessfully(t, GetOutput)

	actual, err := Get(fake.ServiceClient(), "postman_stack", "16ef0584-4458-41eb-87c8-0dc8d5f66c87").Extract()
	th.AssertNoErr(t, err)

	expected := GetExpected
	th.AssertDeepEquals(t, expected, actual)
}

func TestUpdateStack(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleUpdateSuccessfully(t)

	updateOpts := UpdateOpts{
		Template: `
    {
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
          "type":"OS::Nova::Server",
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
	err := Update(fake.ServiceClient(), "gophercloud-test-stack-2", "db6977b2-27aa-4775-9ae7-6213212d4ada", updateOpts).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestUpdateStackNewTemplateFormat(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleUpdateSuccessfully(t)

	template := new(Template)
	template.Bin = []byte(`
		{
			"heat_template_version": "2013-05-23",
			"description": "Simple template to test heat commands",
			"parameters": {
				"flavor": {
					"default": "m1.tiny",
					"type": "string"
				}
			}
		}`)
	updateOpts := UpdateOpts{
		TemplateOpts: template,
	}
	err := Update(fake.ServiceClient(), "gophercloud-test-stack-2", "db6977b2-27aa-4775-9ae7-6213212d4ada", updateOpts).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestDeleteStack(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleDeleteSuccessfully(t)

	err := Delete(fake.ServiceClient(), "gophercloud-test-stack-2", "db6977b2-27aa-4775-9ae7-6213212d4ada").ExtractErr()
	th.AssertNoErr(t, err)
}

func TestPreviewStack(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandlePreviewSuccessfully(t, GetOutput)

	previewOpts := PreviewOpts{
		Name:    "stackcreated",
		Timeout: 60,
		Template: `
    {
      "stack_name": "postman_stack",
      "template": {
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
            "type":"OS::Nova::Server",
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
      }
    }`,
		DisableRollback: Disable,
	}
	actual, err := Preview(fake.ServiceClient(), previewOpts).Extract()
	th.AssertNoErr(t, err)

	expected := PreviewExpected
	th.AssertDeepEquals(t, expected, actual)
}

func TestPreviewStackNewTemplateFormat(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandlePreviewSuccessfully(t, GetOutput)

	template := new(Template)
	template.Bin = []byte(`
		{
			"heat_template_version": "2013-05-23",
			"description": "Simple template to test heat commands",
			"parameters": {
				"flavor": {
					"default": "m1.tiny",
					"type": "string"
				}
			}
		}`)
	previewOpts := PreviewOpts{
		Name:            "stackcreated",
		Timeout:         60,
		TemplateOpts:    template,
		DisableRollback: Disable,
	}
	actual, err := Preview(fake.ServiceClient(), previewOpts).Extract()
	th.AssertNoErr(t, err)

	expected := PreviewExpected
	th.AssertDeepEquals(t, expected, actual)
}

func TestAbandonStack(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	HandleAbandonSuccessfully(t, AbandonOutput)

	actual, err := Abandon(fake.ServiceClient(), "postman_stack", "16ef0584-4458-41eb-87c8-0dc8d5f66c8").Extract()
	th.AssertNoErr(t, err)

	expected := AbandonExpected
	th.AssertDeepEquals(t, expected, actual)
}
