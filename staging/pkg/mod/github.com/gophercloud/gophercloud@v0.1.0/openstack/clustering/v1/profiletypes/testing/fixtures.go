package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/clustering/v1/profiletypes"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

const ProfileTypeRequestID = "req-7328d1b0-9945-456f-b2cd-5166b77d14a8"
const ListResponse = `
{
	"profile_types": [
		{
			"name": "os.nova.server-1.0",
			"schema": {
				"context": {
					"description": "Customized security context for operating containers.",
					"required": false,
					"type": "Map",
					"updatable": false
				},
				"name": {
					"description": "The name of the container.",
					"required": false,
					"type": "Map",
					"updatable": false
				}
			}
		},
		{
			"name": "os.heat.stack-1.0",
			"schema": {
				"context": {
					"default": {},
					"description": "A dictionary for specifying the customized context for stack operations",
					"required": false,
					"type": "Map",
					"updatable": false
				},
				"disable_rollback": {
					"default": true,
					"description": "A boolean specifying whether a stack operation can be rolled back.",
					"required": false,
					"type": "Boolean",
					"updatable": true
				},
				"environment": {
					"default": {},
					"description": "A map that specifies the environment used for stack operations.",
					"required": false,
					"type": "Map",
					"updatable": true
				},
				"files": {
					"default": {},
					"description": "Contents of files referenced by the template, if any.",
					"required": false,
					"type": "Map",
					"updatable": true
				}
			},
			"support_status": {
				"1.0": [
					{
						"status": "SUPPORTED",
						 "since": "2016.04"
					}
				]
			}
		}
	]
}
`

const GetResponse1 = `
{
	"profile_type": {
		"name": "os.nova.server-1.0",
		"schema": {
			"context": {
				"description": "Customized security context for operating containers.",
				"required": false,
				"type": "Map",
				"updatable": false
			},
			"name": {
				"description": "The name of the container.",
				"required": false,
				"type": "Map",
				"updatable": false
			}
		}
	}
}
`

const GetResponse15 = `
{
	"profile_type": {
		"name": "os.heat.stack-1.0",
		"schema": {
			"context": {
				"default": {},
				"description": "A dictionary for specifying the customized context for stack operations",
				"required": false,
				"type": "Map",
				"updatable": false
			},
			"disable_rollback": {
				"default": true,
				"description": "A boolean specifying whether a stack operation can be rolled back.",
				"required": false,
				"type": "Boolean",
				"updatable": true
			},
			"environment": {
				"default": {},
				"description": "A map that specifies the environment used for stack operations.",
				"required": false,
				"type": "Map",
				"updatable": true
			},
			"files": {
				"default": {},
				"description": "Contents of files referenced by the template, if any.",
				"required": false,
				"type": "Map",
				"updatable": true
			}
		},
		"support_status": {
			"1.0": [
				{
					"status": "SUPPORTED",
					"since": "2016.04"
				}
			]
		}
	}
}
`

var ExpectedProfileType1 = profiletypes.ProfileType{
	Name: "os.nova.server-1.0",
	Schema: map[string]profiletypes.Schema{
		"context": {
			"description": "Customized security context for operating containers.",
			"required":    false,
			"type":        "Map",
			"updatable":   false,
		},
		"name": {
			"description": "The name of the container.",
			"required":    false,
			"type":        "Map",
			"updatable":   false,
		},
	},
}

var ExpectedProfileType15 = profiletypes.ProfileType{
	Name: "os.heat.stack-1.0",
	Schema: map[string]profiletypes.Schema{
		"context": {
			"default":     map[string]interface{}{},
			"description": "A dictionary for specifying the customized context for stack operations",
			"required":    false,
			"type":        "Map",
			"updatable":   false,
		},
		"disable_rollback": {
			"default":     true,
			"description": "A boolean specifying whether a stack operation can be rolled back.",
			"required":    false,
			"type":        "Boolean",
			"updatable":   true,
		},
		"environment": {
			"default":     map[string]interface{}{},
			"description": "A map that specifies the environment used for stack operations.",
			"required":    false,
			"type":        "Map",
			"updatable":   true,
		},
		"files": {
			"default":     map[string]interface{}{},
			"description": "Contents of files referenced by the template, if any.",
			"required":    false,
			"type":        "Map",
			"updatable":   true,
		},
	},
	SupportStatus: map[string][]profiletypes.SupportStatus{
		"1.0": {
			{
				"status": "SUPPORTED",
				"since":  "2016.04",
			},
		},
	},
}

var ExpectedProfileTypes = []profiletypes.ProfileType{ExpectedProfileType1, ExpectedProfileType15}

func HandleList1Successfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/profile-types", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, ListResponse)
	})
}

func HandleGet1Successfully(t *testing.T, id string) {
	th.Mux.HandleFunc("/v1/profile-types/"+id, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.Header().Add("X-OpenStack-Request-ID", ProfileTypeRequestID)
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, GetResponse1)
	})
}

func HandleGet15Successfully(t *testing.T, id string) {
	th.Mux.HandleFunc("/v1/profile-types/"+id, func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.Header().Add("X-OpenStack-Request-ID", ProfileTypeRequestID)
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, GetResponse15)
	})
}

const ProfileTypeName = "os.nova.server-1.0"

const ListOpsResponse = `
{
	"operations": {
		"pause": {
			"description": "Pause the server from running.",
			"parameter": null
		},
		"change_password": {
			"description": "Change the administrator password.",
			"parameters": {
				"admin_pass": {
					"description": "New password for the administrator.",
					"required":    false,
					"type":        "String"
				}
			}
		}
	}
}
`

var ExpectedOps = map[string]interface{}{
	"change_password": map[string]interface{}{
		"description": "Change the administrator password.",
		"parameters": map[string]interface{}{
			"admin_pass": map[string]interface{}{
				"description": "New password for the administrator.",
				"required":    false,
				"type":        "String",
			},
		},
	},
	"pause": map[string]interface{}{
		"description": "Pause the server from running.",
		"parameter":   nil,
	},
}

func HandleListOpsSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/profile-types/"+ProfileTypeName+"/ops", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.Header().Add("X-OpenStack-Request-ID", ProfileTypeRequestID)
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, ListOpsResponse)
	})
}
