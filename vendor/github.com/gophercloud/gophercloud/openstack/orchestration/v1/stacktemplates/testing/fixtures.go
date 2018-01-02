package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/orchestration/v1/stacktemplates"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

// GetExpected represents the expected object from a Get request.
var GetExpected = "{\n  \"description\": \"Simple template to test heat commands\",\n  \"heat_template_version\": \"2013-05-23\",\n  \"parameters\": {\n    \"flavor\": {\n      \"default\": \"m1.tiny\",\n      \"type\": \"string\"\n    }\n  },\n  \"resources\": {\n    \"hello_world\": {\n      \"properties\": {\n        \"flavor\": {\n          \"get_param\": \"flavor\"\n        },\n        \"image\": \"ad091b52-742f-469e-8f3c-fd81cadf0743\",\n        \"key_name\": \"heat_key\"\n      },\n      \"type\": \"OS::Nova::Server\"\n    }\n  }\n}"

// GetOutput represents the response body from a Get request.
const GetOutput = `
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
      "type": "OS::Nova::Server",
      "properties": {
        "key_name": "heat_key",
        "flavor": {
          "get_param": "flavor"
        },
        "image": "ad091b52-742f-469e-8f3c-fd81cadf0743"
      }
    }
  }
}`

// HandleGetSuccessfully creates an HTTP handler at `/stacks/postman_stack/16ef0584-4458-41eb-87c8-0dc8d5f66c87/template`
// on the test handler mux that responds with a `Get` response.
func HandleGetSuccessfully(t *testing.T, output string) {
	th.Mux.HandleFunc("/stacks/postman_stack/16ef0584-4458-41eb-87c8-0dc8d5f66c87/template", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, output)
	})
}

// ValidateExpected represents the expected object from a Validate request.
var ValidateExpected = &stacktemplates.ValidatedTemplate{
	Description: "Simple template to test heat commands",
	Parameters: map[string]interface{}{
		"flavor": map[string]interface{}{
			"Default":     "m1.tiny",
			"Type":        "String",
			"NoEcho":      "false",
			"Description": "",
			"Label":       "flavor",
		},
	},
}

// ValidateOutput represents the response body from a Validate request.
const ValidateOutput = `
{
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
}`

// HandleValidateSuccessfully creates an HTTP handler at `/validate`
// on the test handler mux that responds with a `Validate` response.
func HandleValidateSuccessfully(t *testing.T, output string) {
	th.Mux.HandleFunc("/validate", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, output)
	})
}
