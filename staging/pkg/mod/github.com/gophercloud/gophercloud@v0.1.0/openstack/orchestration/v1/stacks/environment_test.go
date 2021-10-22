package stacks

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"testing"

	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestEnvironmentValidation(t *testing.T) {

	environmentJSON := new(Environment)
	environmentJSON.Bin = []byte(ValidJSONEnvironment)
	err := environmentJSON.Validate()
	th.AssertNoErr(t, err)

	environmentYAML := new(Environment)
	environmentYAML.Bin = []byte(ValidYAMLEnvironment)
	err = environmentYAML.Validate()
	th.AssertNoErr(t, err)

	environmentInvalid := new(Environment)
	environmentInvalid.Bin = []byte(InvalidEnvironment)
	if err = environmentInvalid.Validate(); err == nil {
		t.Error("environment validation did not catch invalid environment")
	}
}

func TestEnvironmentParsing(t *testing.T) {
	environmentJSON := new(Environment)
	environmentJSON.Bin = []byte(ValidJSONEnvironment)
	err := environmentJSON.Parse()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ValidJSONEnvironmentParsed, environmentJSON.Parsed)

	environmentYAML := new(Environment)
	environmentYAML.Bin = []byte(ValidJSONEnvironment)
	err = environmentYAML.Parse()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ValidJSONEnvironmentParsed, environmentYAML.Parsed)

	environmentInvalid := new(Environment)
	environmentInvalid.Bin = []byte("Keep Austin Weird")
	err = environmentInvalid.Parse()
	if err == nil {
		t.Error("environment parsing did not catch invalid environment")
	}
}

func TestIgnoreIfEnvironment(t *testing.T) {
	var keyValueTests = []struct {
		key   string
		value interface{}
		out   bool
	}{
		{"base_url", "afksdf", true},
		{"not_type", "hooks", false},
		{"get_file", "::", true},
		{"hooks", "dfsdfsd", true},
		{"type", "sdfubsduf.yaml", false},
		{"type", "sdfsdufs.environment", false},
		{"type", "sdfsdf.file", false},
		{"type", map[string]string{"key": "value"}, true},
	}
	var result bool
	for _, kv := range keyValueTests {
		result = ignoreIfEnvironment(kv.key, kv.value)
		if result != kv.out {
			t.Errorf("key: %v, value: %v expected: %v, actual: %v", kv.key, kv.value, kv.out, result)
		}
	}
}

func TestGetRRFileContents(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	environmentContent := `
heat_template_version: 2013-05-23

description:
  Heat WordPress template to support F18, using only Heat OpenStack-native
  resource types, and without the requirement for heat-cfntools in the image.
  WordPress is web software you can use to create a beautiful website or blog.
  This template installs a single-instance WordPress deployment using a local
  MySQL database to store the data.

parameters:

  key_name:
    type: string
    description : Name of a KeyPair to enable SSH access to the instance

resources:
  wordpress_instance:
    type: OS::Nova::Server
    properties:
      image: { get_param: image_id }
      flavor: { get_param: instance_type }
      key_name: { get_param: key_name }`

	dbContent := `
heat_template_version: 2014-10-16

description:
  Test template for Trove resource capabilities

parameters:
  db_pass:
    type: string
    hidden: true
    description: Database access password
    default: secrete

resources:

service_db:
  type: OS::Trove::Instance
  properties:
    name: trove_test_db
    datastore_type: mariadb
    flavor: 1GB Instance
    size: 10
    databases:
    - name: test_data
    users:
    - name: kitchen_sink
      password: { get_param: db_pass }
      databases: [ test_data ]`
	baseurl, err := getBasePath()
	th.AssertNoErr(t, err)

	fakeEnvURL := strings.Join([]string{baseurl, "my_env.yaml"}, "/")
	urlparsed, err := url.Parse(fakeEnvURL)
	th.AssertNoErr(t, err)
	// handler for my_env.yaml
	th.Mux.HandleFunc(urlparsed.Path, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, environmentContent)
	})

	fakeDBURL := strings.Join([]string{baseurl, "my_db.yaml"}, "/")
	urlparsed, err = url.Parse(fakeDBURL)
	th.AssertNoErr(t, err)

	// handler for my_db.yaml
	th.Mux.HandleFunc(urlparsed.Path, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, dbContent)
	})

	client := fakeClient{BaseClient: getHTTPClient()}
	env := new(Environment)
	env.Bin = []byte(`{"resource_registry": {"My::WP::Server": "my_env.yaml", "resources": {"my_db_server": {"OS::DBInstance": "my_db.yaml"}}}}`)
	env.client = client

	err = env.Parse()
	th.AssertNoErr(t, err)
	err = env.getRRFileContents(ignoreIfEnvironment)
	th.AssertNoErr(t, err)
	expectedEnvFilesContent := "\nheat_template_version: 2013-05-23\n\ndescription:\n  Heat WordPress template to support F18, using only Heat OpenStack-native\n  resource types, and without the requirement for heat-cfntools in the image.\n  WordPress is web software you can use to create a beautiful website or blog.\n  This template installs a single-instance WordPress deployment using a local\n  MySQL database to store the data.\n\nparameters:\n\n  key_name:\n    type: string\n    description : Name of a KeyPair to enable SSH access to the instance\n\nresources:\n  wordpress_instance:\n    type: OS::Nova::Server\n    properties:\n      image: { get_param: image_id }\n      flavor: { get_param: instance_type }\n      key_name: { get_param: key_name }"
	expectedDBFilesContent := "\nheat_template_version: 2014-10-16\n\ndescription:\n  Test template for Trove resource capabilities\n\nparameters:\n  db_pass:\n    type: string\n    hidden: true\n    description: Database access password\n    default: secrete\n\nresources:\n\nservice_db:\n  type: OS::Trove::Instance\n  properties:\n    name: trove_test_db\n    datastore_type: mariadb\n    flavor: 1GB Instance\n    size: 10\n    databases:\n    - name: test_data\n    users:\n    - name: kitchen_sink\n      password: { get_param: db_pass }\n      databases: [ test_data ]"

	th.AssertEquals(t, expectedEnvFilesContent, env.Files[fakeEnvURL])
	th.AssertEquals(t, expectedDBFilesContent, env.Files[fakeDBURL])

	// Update env's fileMaps to replace relative filenames by absolute URLs.
	env.fileMaps = map[string]string{
		"my_env.yaml": fakeEnvURL,
		"my_db.yaml":  fakeDBURL,
	}
	env.fixFileRefs()

	expectedParsed := map[string]interface{}{
		"resource_registry": map[string]interface{}{
			"My::WP::Server": fakeEnvURL,
			"resources": map[string]interface{}{
				"my_db_server": map[string]interface{}{
					"OS::DBInstance": fakeDBURL,
				},
			},
		},
	}
	env.Parse()
	th.AssertDeepEquals(t, expectedParsed, env.Parsed)
}
