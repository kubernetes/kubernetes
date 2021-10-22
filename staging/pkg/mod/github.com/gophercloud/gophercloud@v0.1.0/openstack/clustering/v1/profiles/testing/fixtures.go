package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud/openstack/clustering/v1/profiles"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

const CreateResponse = `
{
  "profile": {
    "created_at": "2016-01-03T16:22:23Z",
    "domain": null,
    "id": "9e1c6f42-acf5-4688-be2c-8ce954ef0f23",
    "metadata": {},
    "name": "test-profile",
    "project": "42d9e9663331431f97b75e25136307ff",
    "spec": {
      "properties": {
        "flavor": "t2.small",
        "image": "centos7.3-latest",
        "name": "centos_server",
        "networks": [
            {
              "network": "private-network"
            }
        ]
      },
      "type": "os.nova.server",
      "version": "1.0"
    },
    "type": "os.nova.server-1.0",
    "updated_at": "2016-01-03T17:22:23Z",
    "user": "5e5bf8027826429c96af157f68dc9072"
  }
}`

var ExpectedCreate = profiles.Profile{
	CreatedAt: time.Date(2016, 1, 3, 16, 22, 23, 0, time.UTC),
	Domain:    "",
	ID:        "9e1c6f42-acf5-4688-be2c-8ce954ef0f23",
	Metadata:  map[string]interface{}{},
	Name:      "test-profile",
	Project:   "42d9e9663331431f97b75e25136307ff",
	Spec: profiles.Spec{
		Properties: map[string]interface{}{
			"flavor": "t2.small",
			"image":  "centos7.3-latest",
			"name":   "centos_server",
			"networks": []interface{}{
				map[string]interface{}{"network": "private-network"},
			},
		},
		Type:    "os.nova.server",
		Version: "1.0",
	},
	Type:      "os.nova.server-1.0",
	UpdatedAt: time.Date(2016, 1, 3, 17, 22, 23, 0, time.UTC),
	User:      "5e5bf8027826429c96af157f68dc9072",
}

const GetResponse = `
{
  "profile": {
    "created_at": "2016-01-03T16:22:23Z",
    "domain": null,
    "id": "9e1c6f42-acf5-4688-be2c-8ce954ef0f23",
    "metadata": {},
    "name": "pserver",
    "project": "42d9e9663331431f97b75e25136307ff",
    "spec": {
      "properties": {
        "flavor": 1,
        "image": "cirros-0.3.4-x86_64-uec",
        "key_name": "oskey",
        "name": "cirros_server",
        "networks": [
          {
            "network": "private"
          }
        ]
      },
      "type": "os.nova.server",
      "version": "1.0"
    },
    "type": "os.nova.server-1.0",
    "updated_at": null,
    "user": "5e5bf8027826429c96af157f68dc9072"
  }
}`

var ExpectedGet = profiles.Profile{
	CreatedAt: time.Date(2016, 1, 3, 16, 22, 23, 0, time.UTC),
	Domain:    "",
	ID:        "9e1c6f42-acf5-4688-be2c-8ce954ef0f23",
	Metadata:  map[string]interface{}{},
	Name:      "pserver",
	Project:   "42d9e9663331431f97b75e25136307ff",
	Spec: profiles.Spec{
		Properties: map[string]interface{}{
			"flavor":   float64(1),
			"image":    "cirros-0.3.4-x86_64-uec",
			"key_name": "oskey",
			"name":     "cirros_server",
			"networks": []interface{}{
				map[string]interface{}{"network": "private"},
			},
		},
		Type:    "os.nova.server",
		Version: "1.0",
	},
	Type: "os.nova.server-1.0",
	User: "5e5bf8027826429c96af157f68dc9072",
}

const ListResponse = `
{
  "profiles": [
    {
      "created_at": "2016-01-03T16:22:23Z",
      "domain": null,
      "id": "9e1c6f42-acf5-4688-be2c-8ce954ef0f23",
      "metadata": {},
      "name": "pserver",
      "project": "42d9e9663331431f97b75e25136307ff",
      "spec": {
        "properties": {
          "flavor": "t2.small",
          "image": "cirros-0.3.4-x86_64-uec",
          "key_name": "oskey",
          "name": "cirros_server",
          "networks": [
            {
              "network": "private"
            }
          ]
        },
        "type": "os.nova.server",
        "version": 1.0
      },
      "type": "os.nova.server-1.0",
      "updated_at": "2016-01-03T17:22:23Z",
      "user": "5e5bf8027826429c96af157f68dc9072"
    },
    {
      "created_at": null,
      "domain": null,
      "id": "9e1c6f42-acf5-4688-be2c-8ce954ef0f23",
      "metadata": {},
      "name": "pserver",
      "project": "42d9e9663331431f97b75e25136307ff",
      "spec": {
        "properties": {
          "flavor": "t2.small",
          "image": "cirros-0.3.4-x86_64-uec",
          "key_name": "oskey",
          "name": "cirros_server",
          "networks": [
            {
              "network": "private"
            }
          ]
        },
        "type": "os.nova.server",
        "version": 1.0
      },
      "type": "os.nova.server-1.0",
      "updated_at": null,
      "user": "5e5bf8027826429c96af157f68dc9072"
    },
    {
      "created_at": "",
      "domain": null,
      "id": "9e1c6f42-acf5-4688-be2c-8ce954ef0f23",
      "metadata": {},
      "name": "pserver",
      "project": "42d9e9663331431f97b75e25136307ff",
      "spec": {
        "properties": {
          "flavor": "t2.small",
          "image": "cirros-0.3.4-x86_64-uec",
          "key_name": "oskey",
          "name": "cirros_server",
          "networks": [
            {
              "network": "private"
            }
          ]
        },
        "type": "os.nova.server",
        "version": "1.0"
      },
      "type": "os.nova.server-1.0",
      "updated_at": "",
      "user": "5e5bf8027826429c96af157f68dc9072"
    }
  ]
}`

var ExpectedListProfile1 = profiles.Profile{
	CreatedAt: time.Date(2016, 1, 3, 16, 22, 23, 0, time.UTC),
	Domain:    "",
	ID:        "9e1c6f42-acf5-4688-be2c-8ce954ef0f23",
	Metadata:  map[string]interface{}{},
	Name:      "pserver",
	Project:   "42d9e9663331431f97b75e25136307ff",
	Spec: profiles.Spec{
		Properties: map[string]interface{}{
			"flavor":   "t2.small",
			"image":    "cirros-0.3.4-x86_64-uec",
			"key_name": "oskey",
			"name":     "cirros_server",
			"networks": []interface{}{
				map[string]interface{}{"network": "private"},
			},
		},
		Type:    "os.nova.server",
		Version: "1.0",
	},
	Type:      "os.nova.server-1.0",
	UpdatedAt: time.Date(2016, 1, 3, 17, 22, 23, 0, time.UTC),
	User:      "5e5bf8027826429c96af157f68dc9072",
}

var ExpectedListProfile2 = profiles.Profile{
	Domain:   "",
	ID:       "9e1c6f42-acf5-4688-be2c-8ce954ef0f23",
	Metadata: map[string]interface{}{},
	Name:     "pserver",
	Project:  "42d9e9663331431f97b75e25136307ff",
	Spec: profiles.Spec{
		Properties: map[string]interface{}{
			"flavor":   "t2.small",
			"image":    "cirros-0.3.4-x86_64-uec",
			"key_name": "oskey",
			"name":     "cirros_server",
			"networks": []interface{}{
				map[string]interface{}{"network": "private"},
			},
		},
		Type:    "os.nova.server",
		Version: "1.0",
	},
	Type: "os.nova.server-1.0",
	User: "5e5bf8027826429c96af157f68dc9072",
}

var ExpectedListProfile3 = profiles.Profile{
	Domain:   "",
	ID:       "9e1c6f42-acf5-4688-be2c-8ce954ef0f23",
	Metadata: map[string]interface{}{},
	Name:     "pserver",
	Project:  "42d9e9663331431f97b75e25136307ff",
	Spec: profiles.Spec{
		Properties: map[string]interface{}{
			"flavor":   "t2.small",
			"image":    "cirros-0.3.4-x86_64-uec",
			"key_name": "oskey",
			"name":     "cirros_server",
			"networks": []interface{}{
				map[string]interface{}{"network": "private"},
			},
		},
		Type:    "os.nova.server",
		Version: "1.0",
	},
	Type: "os.nova.server-1.0",
	User: "5e5bf8027826429c96af157f68dc9072",
}

var ExpectedList = []profiles.Profile{
	ExpectedListProfile1,
	ExpectedListProfile2,
	ExpectedListProfile3,
}

const UpdateResponse = `
{
  "profile": {
    "created_at": "2016-01-03T16:22:23Z",
    "domain": null,
    "id": "9e1c6f42-acf5-4688-be2c-8ce954ef0f23",
    "metadata": {
      "foo": "bar"
    },
    "name": "pserver",
    "project": "42d9e9663331431f97b75e25136307ff",
    "spec": {
      "properties": {
        "flavor": 1,
        "image": "cirros-0.3.4-x86_64-uec",
        "key_name": "oskey",
        "name": "cirros_server",
        "networks": [
          {
            "network": "private"
          }
        ]
      },
      "type": "os.nova.server",
      "version": "1.0"
    },
    "type": "os.nova.server-1.0",
    "updated_at": "2016-01-03T17:22:23Z",
    "user": "5e5bf8027826429c96af157f68dc9072"
  }
}`

var ExpectedUpdate = profiles.Profile{
	CreatedAt: time.Date(2016, 1, 3, 16, 22, 23, 0, time.UTC),
	Domain:    "",
	ID:        "9e1c6f42-acf5-4688-be2c-8ce954ef0f23",
	Metadata:  map[string]interface{}{"foo": "bar"},
	Name:      "pserver",
	Project:   "42d9e9663331431f97b75e25136307ff",
	Spec: profiles.Spec{
		Properties: map[string]interface{}{
			"flavor":   float64(1),
			"image":    "cirros-0.3.4-x86_64-uec",
			"key_name": "oskey",
			"name":     "cirros_server",
			"networks": []interface{}{
				map[string]interface{}{"network": "private"},
			},
		},
		Type:    "os.nova.server",
		Version: "1.0",
	},
	Type:      "os.nova.server-1.0",
	UpdatedAt: time.Date(2016, 1, 3, 17, 22, 23, 0, time.UTC),
	User:      "5e5bf8027826429c96af157f68dc9072",
}

const ValidateResponse = `
{
	"profile": {
		"created_at": "2016-01-03T16:22:23Z",
		"domain": null,
		"id": "9e1c6f42-acf5-4688-be2c-8ce954ef0f23",
		"metadata": {},
		"name": "pserver",
		"project": "42d9e9663331431f97b75e25136307ff",
		"spec": {
			"properties": {
				"flavor": "t2.micro",
				"image": "cirros-0.3.4-x86_64-uec",
				"key_name": "oskey",
				"name": "cirros_server",
				"networks": [
					{
						"network": "private"
					}
				]
			},
			"type": "os.nova.server",
			"version": "1.0"
		},
		"type": "os.nova.server-1.0",
		"updated_at": "2016-01-03T17:22:23Z",
		"user": "5e5bf8027826429c96af157f68dc9072"
	}
}`

var ExpectedValidate = profiles.Profile{
	CreatedAt: time.Date(2016, 1, 3, 16, 22, 23, 0, time.UTC),
	Domain:    "",
	ID:        "9e1c6f42-acf5-4688-be2c-8ce954ef0f23",
	Metadata:  map[string]interface{}{},
	Name:      "pserver",
	Project:   "42d9e9663331431f97b75e25136307ff",
	Spec: profiles.Spec{
		Properties: map[string]interface{}{
			"flavor":   "t2.micro",
			"image":    "cirros-0.3.4-x86_64-uec",
			"key_name": "oskey",
			"name":     "cirros_server",
			"networks": []interface{}{
				map[string]interface{}{"network": "private"},
			},
		},
		Type:    "os.nova.server",
		Version: "1.0",
	},
	Type:      "os.nova.server-1.0",
	UpdatedAt: time.Date(2016, 1, 3, 17, 22, 23, 0, time.UTC),
	User:      "5e5bf8027826429c96af157f68dc9072",
}

func HandleCreateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/profiles", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, CreateResponse)
	})
}

func HandleGetSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/profiles/9e1c6f42-acf5-4688-be2c-8ce954ef0f23", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, GetResponse)
	})
}

func HandleListSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/profiles", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, ListResponse)
	})
}

func HandleUpdateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/profiles/9e1c6f42-acf5-4688-be2c-8ce954ef0f23", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PATCH")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, UpdateResponse)
	})
}

func HandleDeleteSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/profiles/6dc6d336e3fc4c0a951b5698cd1236ee", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusNoContent)
	})
}

func HandleValidateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/profiles/validate", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "OpenStack-API-Version", "clustering 1.2")

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, ValidateResponse)
	})
}
