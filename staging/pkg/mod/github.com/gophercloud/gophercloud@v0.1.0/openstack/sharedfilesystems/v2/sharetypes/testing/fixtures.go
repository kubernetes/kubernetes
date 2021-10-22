package testing

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func MockCreateResponse(t *testing.T) {
	th.Mux.HandleFunc("/types", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
        {
            "share_type": {
                "os-share-type-access:is_public": true,
                "extra_specs": {
                    "driver_handles_share_servers": true,
                    "snapshot_support": true
                },
                "name": "my_new_share_type"
            }
        }`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusAccepted)

		fmt.Fprintf(w, `
        {
            "volume_type": {
                "os-share-type-access:is_public": true,
                "required_extra_specs": {
                    "driver_handles_share_servers": true
                },
                "extra_specs": {
                    "snapshot_support": "True",
                    "driver_handles_share_servers": "True"
                },
                "name": "my_new_share_type",
                "id": "1d600d02-26a7-4b23-af3d-7d51860fe858"
            },
            "share_type": {
                "os-share-type-access:is_public": true,
                "required_extra_specs": {
                    "driver_handles_share_servers": true
                },
                "extra_specs": {
                    "snapshot_support": "True",
                    "driver_handles_share_servers": "True"
                },
                "name": "my_new_share_type",
                "id": "1d600d02-26a7-4b23-af3d-7d51860fe858"
            }
        }`)
	})
}

func MockDeleteResponse(t *testing.T) {
	th.Mux.HandleFunc("/types/shareTypeID", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusAccepted)
	})
}

func MockListResponse(t *testing.T) {
	th.Mux.HandleFunc("/types", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
        {
            "volume_types": [
                {
                    "os-share-type-access:is_public": true,
                    "required_extra_specs": {
                        "driver_handles_share_servers": "True"
                    },
                    "extra_specs": {
                        "snapshot_support": "True",
                        "driver_handles_share_servers": "True"
                    },
                    "name": "default",
                    "id": "be27425c-f807-4500-a056-d00721db45cf"
                },
                {
                    "os-share-type-access:is_public": true,
                    "required_extra_specs": {
                        "driver_handles_share_servers": "false"
                    },
                    "extra_specs": {
                        "snapshot_support": "True",
                        "driver_handles_share_servers": "false"
                    },
                    "name": "d",
                    "id": "f015bebe-c38b-4c49-8832-00143b10253b"
                }
            ],
            "share_types": [
                {
                    "os-share-type-access:is_public": true,
                    "required_extra_specs": {
                        "driver_handles_share_servers": "True"
                    },
                    "extra_specs": {
                        "snapshot_support": "True",
                        "driver_handles_share_servers": "True"
                    },
                    "name": "default",
                    "id": "be27425c-f807-4500-a056-d00721db45cf"
                },
                {
                    "os-share-type-access:is_public": true,
                    "required_extra_specs": {
                        "driver_handles_share_servers": "false"
                    },
                    "extra_specs": {
                        "snapshot_support": "True",
                        "driver_handles_share_servers": "false"
                    },
                    "name": "d",
                    "id": "f015bebe-c38b-4c49-8832-00143b10253b"
                }
            ]
        }`)
	})
}

func MockGetDefaultResponse(t *testing.T) {
	th.Mux.HandleFunc("/types/default", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
        {
            "volume_type": {
                "required_extra_specs": null,
                "extra_specs": {
                    "snapshot_support": "True",
                    "driver_handles_share_servers": "True"
                },
                "name": "default",
                "id": "be27425c-f807-4500-a056-d00721db45cf"
            },
            "share_type": {
                "required_extra_specs": null,
                "extra_specs": {
                    "snapshot_support": "True",
                    "driver_handles_share_servers": "True"
                },
                "name": "default",
                "id": "be27425c-f807-4500-a056-d00721db45cf"
            }
        }`)
	})
}

func MockGetExtraSpecsResponse(t *testing.T) {
	th.Mux.HandleFunc("/types/shareTypeID/extra_specs", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
        {
            "extra_specs": {
                "snapshot_support": "True",
                "driver_handles_share_servers": "True",
                "my_custom_extra_spec": "False"
            }
        }`)
	})
}

func MockSetExtraSpecsResponse(t *testing.T) {
	th.Mux.HandleFunc("/types/shareTypeID/extra_specs", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
        {
            "extra_specs": {
                "my_key": "my_value"
            }
        }`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusAccepted)

		fmt.Fprintf(w, `
        {
            "extra_specs": {
                "my_key": "my_value"
            }
        }`)
	})
}

func MockUnsetExtraSpecsResponse(t *testing.T) {
	th.Mux.HandleFunc("/types/shareTypeID/extra_specs/my_key", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusAccepted)
	})
}

func MockShowAccessResponse(t *testing.T) {
	th.Mux.HandleFunc("/types/shareTypeID/share_type_access", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
        {
            "share_type_access": [
                {
                    "share_type_id": "1732f284-401d-41d9-a494-425451e8b4b8",
                    "project_id": "818a3f48dcd644909b3fa2e45a399a27"
                },
                {
                    "share_type_id": "1732f284-401d-41d9-a494-425451e8b4b8",
                    "project_id": "e1284adea3ee4d2482af5ed214f3ad90"
                }
            ]
        }`)
	})
}

func MockAddAccessResponse(t *testing.T) {
	th.Mux.HandleFunc("/types/shareTypeID/action", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
        {
            "addProjectAccess": {
                "project": "e1284adea3ee4d2482af5ed214f3ad90"
            }
        }`)
		w.WriteHeader(http.StatusAccepted)
	})
}

func MockRemoveAccessResponse(t *testing.T) {
	th.Mux.HandleFunc("/types/shareTypeID/action", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
        {
            "removeProjectAccess": {
                "project": "e1284adea3ee4d2482af5ed214f3ad90"
            }
        }`)
		w.WriteHeader(http.StatusAccepted)
	})
}
