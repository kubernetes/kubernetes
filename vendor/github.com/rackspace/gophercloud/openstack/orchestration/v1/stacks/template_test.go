package stacks

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
)

func TestTemplateValidation(t *testing.T) {
	templateJSON := new(Template)
	templateJSON.Bin = []byte(ValidJSONTemplate)
	err := templateJSON.Validate()
	th.AssertNoErr(t, err)

	templateYAML := new(Template)
	templateYAML.Bin = []byte(ValidYAMLTemplate)
	err = templateYAML.Validate()
	th.AssertNoErr(t, err)

	templateInvalid := new(Template)
	templateInvalid.Bin = []byte(InvalidTemplateNoVersion)
	if err = templateInvalid.Validate(); err == nil {
		t.Error("Template validation did not catch invalid template")
	}
}

func TestTemplateParsing(t *testing.T) {
	templateJSON := new(Template)
	templateJSON.Bin = []byte(ValidJSONTemplate)
	err := templateJSON.Parse()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ValidJSONTemplateParsed, templateJSON.Parsed)

	templateYAML := new(Template)
	templateYAML.Bin = []byte(ValidJSONTemplate)
	err = templateYAML.Parse()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ValidJSONTemplateParsed, templateYAML.Parsed)

	templateInvalid := new(Template)
	templateInvalid.Bin = []byte("Keep Austin Weird")
	err = templateInvalid.Parse()
	if err == nil {
		t.Error("Template parsing did not catch invalid template")
	}
}

func TestIgnoreIfTemplate(t *testing.T) {
	var keyValueTests = []struct {
		key   string
		value interface{}
		out   bool
	}{
		{"not_get_file", "afksdf", true},
		{"not_type", "sdfd", true},
		{"get_file", "shdfuisd", false},
		{"type", "dfsdfsd", true},
		{"type", "sdfubsduf.yaml", false},
		{"type", "sdfsdufs.template", false},
		{"type", "sdfsdf.file", true},
		{"type", map[string]string{"key": "value"}, true},
	}
	var result bool
	for _, kv := range keyValueTests {
		result = ignoreIfTemplate(kv.key, kv.value)
		if result != kv.out {
			t.Errorf("key: %v, value: %v expected: %v, actual: %v", kv.key, kv.value, result, kv.out)
		}
	}
}

func TestGetFileContents(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	baseurl, err := getBasePath()
	th.AssertNoErr(t, err)
	fakeURL := strings.Join([]string{baseurl, "my_nova.yaml"}, "/")
	urlparsed, err := url.Parse(fakeURL)
	th.AssertNoErr(t, err)
	myNovaContent := `heat_template_version: 2014-10-16
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
      networks:
      - {uuid: 11111111-1111-1111-1111-111111111111}`
	th.Mux.HandleFunc(urlparsed.Path, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		w.Header().Set("Content-Type", "application/jason")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, myNovaContent)
	})

	client := fakeClient{BaseClient: getHTTPClient()}
	te := new(Template)
	te.Bin = []byte(`heat_template_version: 2015-04-30
resources:
  my_server:
    type: my_nova.yaml`)
	te.client = client

	err = te.Parse()
	th.AssertNoErr(t, err)
	err = te.getFileContents(te.Parsed, ignoreIfTemplate, true)
	th.AssertNoErr(t, err)
	expectedFiles := map[string]string{
		"my_nova.yaml": `heat_template_version: 2014-10-16
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
      networks:
      - {uuid: 11111111-1111-1111-1111-111111111111}`}
	th.AssertEquals(t, expectedFiles["my_nova.yaml"], te.Files[fakeURL])
	te.fixFileRefs()
	expectedParsed := map[string]interface{}{
		"heat_template_version": "2015-04-30",
		"resources": map[string]interface{}{
			"my_server": map[string]interface{}{
				"type": fakeURL,
			},
		},
	}
	te.Parse()
	th.AssertDeepEquals(t, expectedParsed, te.Parsed)
}
