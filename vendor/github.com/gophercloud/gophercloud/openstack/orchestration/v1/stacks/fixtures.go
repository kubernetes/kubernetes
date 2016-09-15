package stacks

// ValidJSONTemplate is a valid OpenStack Heat template in JSON format
const ValidJSONTemplate = `
{
  "heat_template_version": "2014-10-16",
  "parameters": {
    "flavor": {
      "default": 4353,
      "description": "Flavor for the server to be created",
      "hidden": true,
      "type": "string"
    }
  },
  "resources": {
    "test_server": {
      "properties": {
        "flavor": "2 GB General Purpose v1",
        "image": "Debian 7 (Wheezy) (PVHVM)",
        "name": "test-server"
      },
      "type": "OS::Nova::Server"
    }
  }
}
`

// ValidYAMLTemplate is a valid OpenStack Heat template in YAML format
const ValidYAMLTemplate = `
heat_template_version: 2014-10-16
parameters:
  flavor:
    type: string
    description: Flavor for the server to be created
    default: 4353
    hidden: true
resources:
  test_server:
    type: "OS::Nova::Server"
    properties:
      name: test-server
      flavor: 2 GB General Purpose v1
      image: Debian 7 (Wheezy) (PVHVM)
`

// InvalidTemplateNoVersion is an invalid template as it has no `version` section
const InvalidTemplateNoVersion = `
parameters:
  flavor:
    type: string
    description: Flavor for the server to be created
    default: 4353
    hidden: true
resources:
  test_server:
    type: "OS::Nova::Server"
    properties:
      name: test-server
      flavor: 2 GB General Purpose v1
      image: Debian 7 (Wheezy) (PVHVM)
`

// ValidJSONEnvironment is a valid environment for a stack in JSON format
const ValidJSONEnvironment = `
{
	"parameters": {
		"user_key": "userkey"
	},
	"resource_registry": {
		"My::WP::Server": "file:///home/shardy/git/heat-templates/hot/F18/WordPress_Native.yaml",
		"OS::Quantum*": "OS::Neutron*",
		"AWS::CloudWatch::Alarm": "file:///etc/heat/templates/AWS_CloudWatch_Alarm.yaml",
		"OS::Metering::Alarm": "OS::Ceilometer::Alarm",
		"AWS::RDS::DBInstance": "file:///etc/heat/templates/AWS_RDS_DBInstance.yaml",
		"resources": {
			"my_db_server": {
				"OS::DBInstance": "file:///home/mine/all_my_cool_templates/db.yaml"
			},
			"my_server": {
				"OS::DBInstance": "file:///home/mine/all_my_cool_templates/db.yaml",
				"hooks": "pre-create"
			},
			"nested_stack": {
				"nested_resource": {
					"hooks": "pre-update"
				},
				"another_resource": {
					"hooks": [
						"pre-create",
						"pre-update"
					]
				}
			}
		}
	}
}
`

// ValidYAMLEnvironment is a valid environment for a stack in YAML format
const ValidYAMLEnvironment = `
parameters:
  user_key: userkey
resource_registry:
  My::WP::Server: file:///home/shardy/git/heat-templates/hot/F18/WordPress_Native.yaml
  # allow older templates with Quantum in them.
  "OS::Quantum*": "OS::Neutron*"
  # Choose your implementation of AWS::CloudWatch::Alarm
  "AWS::CloudWatch::Alarm": "file:///etc/heat/templates/AWS_CloudWatch_Alarm.yaml"
  #"AWS::CloudWatch::Alarm": "OS::Heat::CWLiteAlarm"
  "OS::Metering::Alarm": "OS::Ceilometer::Alarm"
  "AWS::RDS::DBInstance": "file:///etc/heat/templates/AWS_RDS_DBInstance.yaml"
  resources:
    my_db_server:
      "OS::DBInstance": file:///home/mine/all_my_cool_templates/db.yaml
    my_server:
      "OS::DBInstance": file:///home/mine/all_my_cool_templates/db.yaml
      hooks: pre-create
    nested_stack:
      nested_resource:
        hooks: pre-update
      another_resource:
        hooks: [pre-create, pre-update]
`

// InvalidEnvironment is an invalid environment as it has an extra section called `resources`
const InvalidEnvironment = `
parameters:
	flavor:
		type: string
		description: Flavor for the server to be created
		default: 4353
		hidden: true
resources:
	test_server:
		type: "OS::Nova::Server"
		properties:
			name: test-server
			flavor: 2 GB General Purpose v1
			image: Debian 7 (Wheezy) (PVHVM)
parameter_defaults:
	KeyName: heat_key
`

// ValidJSONEnvironmentParsed is the expected parsed version of ValidJSONEnvironment
var ValidJSONEnvironmentParsed = map[string]interface{}{
	"parameters": map[string]interface{}{
		"user_key": "userkey",
	},
	"resource_registry": map[string]interface{}{
		"My::WP::Server":         "file:///home/shardy/git/heat-templates/hot/F18/WordPress_Native.yaml",
		"OS::Quantum*":           "OS::Neutron*",
		"AWS::CloudWatch::Alarm": "file:///etc/heat/templates/AWS_CloudWatch_Alarm.yaml",
		"OS::Metering::Alarm":    "OS::Ceilometer::Alarm",
		"AWS::RDS::DBInstance":   "file:///etc/heat/templates/AWS_RDS_DBInstance.yaml",
		"resources": map[string]interface{}{
			"my_db_server": map[string]interface{}{
				"OS::DBInstance": "file:///home/mine/all_my_cool_templates/db.yaml",
			},
			"my_server": map[string]interface{}{
				"OS::DBInstance": "file:///home/mine/all_my_cool_templates/db.yaml",
				"hooks":          "pre-create",
			},
			"nested_stack": map[string]interface{}{
				"nested_resource": map[string]interface{}{
					"hooks": "pre-update",
				},
				"another_resource": map[string]interface{}{
					"hooks": []interface{}{
						"pre-create",
						"pre-update",
					},
				},
			},
		},
	},
}

// ValidJSONTemplateParsed is the expected parsed version of ValidJSONTemplate
var ValidJSONTemplateParsed = map[string]interface{}{
	"heat_template_version": "2014-10-16",
	"parameters": map[string]interface{}{
		"flavor": map[string]interface{}{
			"default":     4353,
			"description": "Flavor for the server to be created",
			"hidden":      true,
			"type":        "string",
		},
	},
	"resources": map[string]interface{}{
		"test_server": map[string]interface{}{
			"properties": map[string]interface{}{
				"flavor": "2 GB General Purpose v1",
				"image":  "Debian 7 (Wheezy) (PVHVM)",
				"name":   "test-server",
			},
			"type": "OS::Nova::Server",
		},
	},
}
