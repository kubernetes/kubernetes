package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/orchestration/v1/stackresources"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

var Create_time, _ = time.Parse(time.RFC3339, "2018-06-26T07:57:17Z")
var Updated_time, _ = time.Parse(time.RFC3339, "2018-06-26T07:58:17Z")

// FindExpected represents the expected object from a Find request.
var FindExpected = []stackresources.Resource{
	{
		Name: "hello_world",
		Links: []gophercloud.Link{
			{
				Href: "http://166.78.160.107:8004/v1/98606384f58d4ad0b3db7d0d779549ac/stacks/postman_stack/5f57cff9-93fc-424e-9f78-df0515e7f48b/resources/hello_world",
				Rel:  "self",
			},
			{
				Href: "http://166.78.160.107:8004/v1/98606384f58d4ad0b3db7d0d779549ac/stacks/postman_stack/5f57cff9-93fc-424e-9f78-df0515e7f48b",
				Rel:  "stack",
			},
		},
		LogicalID:    "hello_world",
		StatusReason: "state changed",
		UpdatedTime:  Updated_time,
		CreationTime: Create_time,
		RequiredBy:   []interface{}{},
		Status:       "CREATE_IN_PROGRESS",
		PhysicalID:   "49181cd6-169a-4130-9455-31185bbfc5bf",
		Type:         "OS::Nova::Server",
		Attributes:   map[string]interface{}{"SXSW": "atx"},
		Description:  "Some resource",
	},
}

// FindOutput represents the response body from a Find request.
const FindOutput = `
{
  "resources": [
  {
  	"description": "Some resource",
  	"attributes": {"SXSW": "atx"},
    "resource_name": "hello_world",
    "links": [
      {
      "href": "http://166.78.160.107:8004/v1/98606384f58d4ad0b3db7d0d779549ac/stacks/postman_stack/5f57cff9-93fc-424e-9f78-df0515e7f48b/resources/hello_world",
      "rel": "self"
      },
      {
        "href": "http://166.78.160.107:8004/v1/98606384f58d4ad0b3db7d0d779549ac/stacks/postman_stack/5f57cff9-93fc-424e-9f78-df0515e7f48b",
        "rel": "stack"
      }
    ],
    "logical_resource_id": "hello_world",
    "resource_status_reason": "state changed",
    "updated_time": "2018-06-26T07:58:17Z",
	"creation_time": "2018-06-26T07:57:17Z",
    "required_by": [],
    "resource_status": "CREATE_IN_PROGRESS",
    "physical_resource_id": "49181cd6-169a-4130-9455-31185bbfc5bf",
    "resource_type": "OS::Nova::Server"
  }
  ]
}`

// HandleFindSuccessfully creates an HTTP handler at `/stacks/hello_world/resources`
// on the test handler mux that responds with a `Find` response.
func HandleFindSuccessfully(t *testing.T, output string) {
	th.Mux.HandleFunc("/stacks/hello_world/resources", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, output)
	})
}

// ListExpected represents the expected object from a List request.
var ListExpected = []stackresources.Resource{
	{
		Name: "hello_world",
		Links: []gophercloud.Link{
			{
				Href: "http://166.78.160.107:8004/v1/98606384f58d4ad0b3db7d0d779549ac/stacks/postman_stack/5f57cff9-93fc-424e-9f78-df0515e7f48b/resources/hello_world",
				Rel:  "self",
			},
			{
				Href: "http://166.78.160.107:8004/v1/98606384f58d4ad0b3db7d0d779549ac/stacks/postman_stack/5f57cff9-93fc-424e-9f78-df0515e7f48b",
				Rel:  "stack",
			},
		},
		LogicalID:    "hello_world",
		StatusReason: "state changed",
		UpdatedTime:  Updated_time,
		CreationTime: Create_time,
		RequiredBy:   []interface{}{},
		Status:       "CREATE_IN_PROGRESS",
		PhysicalID:   "49181cd6-169a-4130-9455-31185bbfc5bf",
		Type:         "OS::Nova::Server",
		Attributes:   map[string]interface{}{"SXSW": "atx"},
		Description:  "Some resource",
	},
}

// ListOutput represents the response body from a List request.
const ListOutput = `{
  "resources": [
  {
    "resource_name": "hello_world",
    "links": [
    {
      "href": "http://166.78.160.107:8004/v1/98606384f58d4ad0b3db7d0d779549ac/stacks/postman_stack/5f57cff9-93fc-424e-9f78-df0515e7f48b/resources/hello_world",
      "rel": "self"
    },
    {
      "href": "http://166.78.160.107:8004/v1/98606384f58d4ad0b3db7d0d779549ac/stacks/postman_stack/5f57cff9-93fc-424e-9f78-df0515e7f48b",
      "rel": "stack"
    }
    ],
    "logical_resource_id": "hello_world",
    "resource_status_reason": "state changed",
    "updated_time": "2018-06-26T07:58:17Z",
    "creation_time": "2018-06-26T07:57:17Z",
    "required_by": [],
    "resource_status": "CREATE_IN_PROGRESS",
    "physical_resource_id": "49181cd6-169a-4130-9455-31185bbfc5bf",
    "resource_type": "OS::Nova::Server",
	"attributes": {"SXSW": "atx"},
	"description": "Some resource"
  }
]
}`

// HandleListSuccessfully creates an HTTP handler at `/stacks/hello_world/49181cd6-169a-4130-9455-31185bbfc5bf/resources`
// on the test handler mux that responds with a `List` response.
func HandleListSuccessfully(t *testing.T, output string) {
	th.Mux.HandleFunc("/stacks/hello_world/49181cd6-169a-4130-9455-31185bbfc5bf/resources", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		r.ParseForm()
		marker := r.Form.Get("marker")
		switch marker {
		case "":
			fmt.Fprintf(w, output)
		case "49181cd6-169a-4130-9455-31185bbfc5bf":
			fmt.Fprintf(w, `{"resources":[]}`)
		default:
			t.Fatalf("Unexpected marker: [%s]", marker)
		}
	})
}

// GetExpected represents the expected object from a Get request.
var GetExpected = &stackresources.Resource{
	Name: "wordpress_instance",
	Links: []gophercloud.Link{
		{
			Href: "http://166.78.160.107:8004/v1/98606384f58d4ad0b3db7d0d779549ac/stacks/teststack/0b1771bd-9336-4f2b-ae86-a80f971faf1e/resources/wordpress_instance",
			Rel:  "self",
		},
		{
			Href: "http://166.78.160.107:8004/v1/98606384f58d4ad0b3db7d0d779549ac/stacks/teststack/0b1771bd-9336-4f2b-ae86-a80f971faf1e",
			Rel:  "stack",
		},
	},
	LogicalID:    "wordpress_instance",
	Attributes:   map[string]interface{}{"SXSW": "atx"},
	StatusReason: "state changed",
	UpdatedTime:  Updated_time,
	RequiredBy:   []interface{}{},
	Status:       "CREATE_COMPLETE",
	PhysicalID:   "00e3a2fe-c65d-403c-9483-4db9930dd194",
	Type:         "OS::Nova::Server",
}

// GetOutput represents the response body from a Get request.
const GetOutput = `
{
  "resource": {
    "description": "Some resource",
    "attributes": {"SXSW": "atx"},
    "resource_name": "wordpress_instance",
    "description": "",
    "links": [
    {
      "href": "http://166.78.160.107:8004/v1/98606384f58d4ad0b3db7d0d779549ac/stacks/teststack/0b1771bd-9336-4f2b-ae86-a80f971faf1e/resources/wordpress_instance",
      "rel": "self"
    },
    {
      "href": "http://166.78.160.107:8004/v1/98606384f58d4ad0b3db7d0d779549ac/stacks/teststack/0b1771bd-9336-4f2b-ae86-a80f971faf1e",
      "rel": "stack"
    }
    ],
    "logical_resource_id": "wordpress_instance",
    "resource_status": "CREATE_COMPLETE",
    "updated_time": "2018-06-26T07:58:17Z",
    "required_by": [],
    "resource_status_reason": "state changed",
    "physical_resource_id": "00e3a2fe-c65d-403c-9483-4db9930dd194",
    "resource_type": "OS::Nova::Server"
  }
}`

// HandleGetSuccessfully creates an HTTP handler at `/stacks/teststack/0b1771bd-9336-4f2b-ae86-a80f971faf1e/resources/wordpress_instance`
// on the test handler mux that responds with a `Get` response.
func HandleGetSuccessfully(t *testing.T, output string) {
	th.Mux.HandleFunc("/stacks/teststack/0b1771bd-9336-4f2b-ae86-a80f971faf1e/resources/wordpress_instance", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, output)
	})
}

// MetadataExpected represents the expected object from a Metadata request.
var MetadataExpected = map[string]string{
	"number": "7",
	"animal": "auk",
}

// MetadataOutput represents the response body from a Metadata request.
const MetadataOutput = `
{
    "metadata": {
      "number": "7",
      "animal": "auk"
    }
}`

// HandleMetadataSuccessfully creates an HTTP handler at `/stacks/teststack/0b1771bd-9336-4f2b-ae86-a80f971faf1e/resources/wordpress_instance/metadata`
// on the test handler mux that responds with a `Metadata` response.
func HandleMetadataSuccessfully(t *testing.T, output string) {
	th.Mux.HandleFunc("/stacks/teststack/0b1771bd-9336-4f2b-ae86-a80f971faf1e/resources/wordpress_instance/metadata", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, output)
	})
}

// ListTypesExpected represents the expected object from a ListTypes request.
var ListTypesExpected = stackresources.ResourceTypes{
	"OS::Nova::Server",
	"OS::Heat::RandomString",
	"OS::Swift::Container",
	"OS::Trove::Instance",
	"OS::Nova::FloatingIPAssociation",
	"OS::Cinder::VolumeAttachment",
	"OS::Nova::FloatingIP",
	"OS::Nova::KeyPair",
}

// same as above, but sorted
var SortedListTypesExpected = stackresources.ResourceTypes{
	"OS::Cinder::VolumeAttachment",
	"OS::Heat::RandomString",
	"OS::Nova::FloatingIP",
	"OS::Nova::FloatingIPAssociation",
	"OS::Nova::KeyPair",
	"OS::Nova::Server",
	"OS::Swift::Container",
	"OS::Trove::Instance",
}

// ListTypesOutput represents the response body from a ListTypes request.
const ListTypesOutput = `
{
  "resource_types": [
    "OS::Nova::Server",
    "OS::Heat::RandomString",
    "OS::Swift::Container",
    "OS::Trove::Instance",
    "OS::Nova::FloatingIPAssociation",
    "OS::Cinder::VolumeAttachment",
    "OS::Nova::FloatingIP",
    "OS::Nova::KeyPair"
  ]
}`

// HandleListTypesSuccessfully creates an HTTP handler at `/resource_types`
// on the test handler mux that responds with a `ListTypes` response.
func HandleListTypesSuccessfully(t *testing.T, output string) {
	th.Mux.HandleFunc("/resource_types", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, output)
	})
}

// GetSchemaExpected represents the expected object from a Schema request.
var GetSchemaExpected = &stackresources.TypeSchema{
	Attributes: map[string]interface{}{
		"an_attribute": map[string]interface{}{
			"description": "An attribute description .",
		},
	},
	Properties: map[string]interface{}{
		"a_property": map[string]interface{}{
			"update_allowed": false,
			"required":       true,
			"type":           "string",
			"description":    "A resource description.",
		},
	},
	ResourceType: "OS::Heat::AResourceName",
	SupportStatus: map[string]interface{}{
		"message": "A status message",
		"status":  "SUPPORTED",
		"version": "2014.1",
	},
}

// GetSchemaOutput represents the response body from a Schema request.
const GetSchemaOutput = `
{
  "attributes": {
    "an_attribute": {
      "description": "An attribute description ."
    }
  },
  "properties": {
    "a_property": {
      "update_allowed": false,
      "required": true,
      "type": "string",
      "description": "A resource description."
    }
  },
  "resource_type": "OS::Heat::AResourceName",
  "support_status": {
	"message": "A status message",
	"status": "SUPPORTED",
	"version": "2014.1"
  }
}`

// HandleGetSchemaSuccessfully creates an HTTP handler at `/resource_types/OS::Heat::AResourceName`
// on the test handler mux that responds with a `Schema` response.
func HandleGetSchemaSuccessfully(t *testing.T, output string) {
	th.Mux.HandleFunc("/resource_types/OS::Heat::AResourceName", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, output)
	})
}

// GetTemplateExpected represents the expected object from a Template request.
var GetTemplateExpected = "{\n  \"HeatTemplateFormatVersion\": \"2012-12-12\",\n  \"Outputs\": {\n    \"private_key\": {\n      \"Description\": \"The private key if it has been saved.\",\n      \"Value\": \"{\\\"Fn::GetAtt\\\": [\\\"KeyPair\\\", \\\"private_key\\\"]}\"\n    },\n    \"public_key\": {\n      \"Description\": \"The public key.\",\n      \"Value\": \"{\\\"Fn::GetAtt\\\": [\\\"KeyPair\\\", \\\"public_key\\\"]}\"\n    }\n  },\n  \"Parameters\": {\n    \"name\": {\n      \"Description\": \"The name of the key pair.\",\n      \"Type\": \"String\"\n    },\n    \"public_key\": {\n      \"Description\": \"The optional public key. This allows users to supply the public key from a pre-existing key pair. If not supplied, a new key pair will be generated.\",\n      \"Type\": \"String\"\n    },\n    \"save_private_key\": {\n      \"AllowedValues\": [\n        \"True\",\n        \"true\",\n        \"False\",\n        \"false\"\n      ],\n      \"Default\": false,\n      \"Description\": \"True if the system should remember a generated private key; False otherwise.\",\n      \"Type\": \"String\"\n    }\n  },\n  \"Resources\": {\n    \"KeyPair\": {\n      \"Properties\": {\n        \"name\": {\n          \"Ref\": \"name\"\n        },\n        \"public_key\": {\n          \"Ref\": \"public_key\"\n        },\n        \"save_private_key\": {\n          \"Ref\": \"save_private_key\"\n        }\n      },\n      \"Type\": \"OS::Nova::KeyPair\"\n    }\n  }\n}"

// GetTemplateOutput represents the response body from a Template request.
const GetTemplateOutput = `
{
  "HeatTemplateFormatVersion": "2012-12-12",
  "Outputs": {
    "private_key": {
      "Description": "The private key if it has been saved.",
      "Value": "{\"Fn::GetAtt\": [\"KeyPair\", \"private_key\"]}"
    },
    "public_key": {
      "Description": "The public key.",
      "Value": "{\"Fn::GetAtt\": [\"KeyPair\", \"public_key\"]}"
    }
  },
  "Parameters": {
    "name": {
      "Description": "The name of the key pair.",
      "Type": "String"
    },
    "public_key": {
      "Description": "The optional public key. This allows users to supply the public key from a pre-existing key pair. If not supplied, a new key pair will be generated.",
      "Type": "String"
    },
    "save_private_key": {
      "AllowedValues": [
      "True",
      "true",
      "False",
      "false"
      ],
      "Default": false,
      "Description": "True if the system should remember a generated private key; False otherwise.",
      "Type": "String"
    }
  },
  "Resources": {
    "KeyPair": {
      "Properties": {
        "name": {
          "Ref": "name"
        },
        "public_key": {
          "Ref": "public_key"
        },
        "save_private_key": {
          "Ref": "save_private_key"
        }
      },
      "Type": "OS::Nova::KeyPair"
    }
  }
}`

// HandleGetTemplateSuccessfully creates an HTTP handler at `/resource_types/OS::Heat::AResourceName/template`
// on the test handler mux that responds with a `Template` response.
func HandleGetTemplateSuccessfully(t *testing.T, output string) {
	th.Mux.HandleFunc("/resource_types/OS::Heat::AResourceName/template", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, output)
	})
}

// HandleMarkUnhealthySuccessfully creates an HTTP handler at `/stacks/teststack/0b1771bd-9336-4f2b-ae86-a80f971faf1e/resources/wordpress_instance`
// on the test handler mux that responds with a `MarkUnhealthy` response.
func HandleMarkUnhealthySuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/stacks/teststack/0b1771bd-9336-4f2b-ae86-a80f971faf1e/resources/wordpress_instance", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PATCH")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
	})
}
