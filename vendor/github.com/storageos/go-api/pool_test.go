package storageos

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"reflect"
	"testing"

	"github.com/storageos/go-api/types"
)

func TestPoolList(t *testing.T) {
	poolsData := `[
    {
        "active": true,
        "capacity_stats": {
            "available_capacity_bytes": 104236191744,
            "provisioned_capacity_bytes": 5368709120,
            "total_capacity_bytes": 140226224128
        },
        "controller_names": [
            "storageos-1",
            "storageos-2",
            "storageos-3"
        ],
        "default": true,
        "default_driver": "filesystem",
        "description": "Default storage pool",
        "driver_capacity_stats": null,
        "driver_instances": [
            {
                "active": true,
                "capacity_stats": {
                    "available_capacity_bytes": 80296787968,
                    "provisioned_capacity_bytes": 5368709120,
                    "total_capacity_bytes": 103440351232
                },
                "config": {
                    "data_dir": "/var/lib/storageos/data"
                },
                "controller_name": "storageos-1",
                "description": "Default storage pool",
                "driver_name": "filesystem",
                "id": "2935b1b9-a8af-121c-9e79-a64c637f0ee9",
                "name": "default",
                "pool_id": "b4c87d6c-2958-6283-128b-f767153938ad",
								"labels": {
				            "storageos.driver": "filesystem"
				        }
            },
            {
                "active": true,
                "capacity_stats": {
                    "available_capacity_bytes": 9687187456,
                    "provisioned_capacity_bytes": 0,
                    "total_capacity_bytes": 18392936448
                },
                "config": {
                    "data_dir": "/var/lib/storageos/data"
                },
                "controller_name": "storageos-2",
                "description": "Default storage pool",
                "driver_name": "filesystem",
                "id": "b8c8cf01-98a2-0270-6393-e86427103275",
                "name": "default",
                "pool_id": "b4c87d6c-2958-6283-128b-f767153938ad",
								"labels": {
				            "storageos.driver": "filesystem"
				        }
            },
            {
                "active": true,
                "capacity_stats": {
                    "available_capacity_bytes": 14252216320,
                    "provisioned_capacity_bytes": 0,
                    "total_capacity_bytes": 18392936448
                },
                "config": {
                    "data_dir": "/var/lib/storageos/data"
                },
                "controller_name": "storageos-3",
                "description": "Default storage pool",
                "driver_name": "filesystem",
                "id": "097a3e9a-86f1-1c7f-f874-90602b498aff",
                "name": "default",
                "pool_id": "b4c87d6c-2958-6283-128b-f767153938ad",
								"labels": {
				            "storageos.driver": "filesystem"
				        }
            }
        ],
        "driver_names": [
            "filesystem"
        ],
        "id": "b4c87d6c-2958-6283-128b-f767153938ad",
        "labels": null,
        "name": "default",
				"labels": {
            "storageos.driver": "filesystem"
        },
        "tenant": ""
    }
]`

	var expected []*types.Pool
	if err := json.Unmarshal([]byte(poolsData), &expected); err != nil {
		t.Fatal(err)
	}

	client := newTestClient(&FakeRoundTripper{message: poolsData, status: http.StatusOK})
	pools, err := client.PoolList(types.ListOptions{})
	if err != nil {
		t.Error(err)
	}
	if !reflect.DeepEqual(pools, expected) {
		t.Errorf("Pools: Wrong return value. Want %#v. Got %#v.", expected, pools)
	}
}

func TestPoolCreate(t *testing.T) {
	body := `{
		"id": "b4c87d6c-2958-6283-128b-f767153938ad",
		"datacentre": "",
		"tenant": "",
		"name": "default",
		"description": "Default storage pool",
		"default": true,
		"default_driver": "filesystem",
		"active": true,
		"controller_names": [
			"storageos-1",
			"storageos-2",
			"storageos-3"
		],
		"driver_names": [
			"filesystem"
		],
		"driver_instances": [
			{
				"id": "2935b1b9-a8af-121c-9e79-a64c637f0ee9",
				"datacentre": "",
				"tenant": "",
				"name": "default",
				"description": "Default storage pool",
				"type": "",
				"active": true,
				"config": {
					"data_dir": "/var/lib/storageos/data"
				},
				"tags": [
					"prod",
					"london"
				],
				"controller_name": "storageos-1",
				"pool_id": "b4c87d6c-2958-6283-128b-f767153938ad",
				"driver_name": "filesystem",
				"capacity_stats": {
					"total_capacity_bytes": 103440351232,
					"available_capacity_bytes": 76249460736,
					"provisioned_capacity_bytes": 0
				}
			},
			{
				"id": "b8c8cf01-98a2-0270-6393-e86427103275",
				"datacentre": "",
				"tenant": "",
				"name": "default",
				"description": "Default storage pool",
				"type": "",
				"active": true,
				"config": {
					"data_dir": "/var/lib/storageos/data"
				},
				"tags": [
					"prod",
					"london"
				],
				"controller_name": "storageos-2",
				"pool_id": "b4c87d6c-2958-6283-128b-f767153938ad",
				"driver_name": "filesystem",
				"capacity_stats": {
					"total_capacity_bytes": 18392936448,
					"available_capacity_bytes": 5679550464,
					"provisioned_capacity_bytes": 0
				}
			},
			{
				"id": "097a3e9a-86f1-1c7f-f874-90602b498aff",
				"datacentre": "",
				"tenant": "",
				"name": "default",
				"description": "Default storage pool",
				"type": "",
				"active": true,
				"config": {
					"data_dir": "/var/lib/storageos/data"
				},
				"tags": [
					"prod",
					"london"
				],
				"controller_name": "storageos-3",
				"pool_id": "b4c87d6c-2958-6283-128b-f767153938ad",
				"driver_name": "filesystem",
				"capacity_stats": {
					"total_capacity_bytes": 18392936448,
					"available_capacity_bytes": 10806575104,
					"provisioned_capacity_bytes": 0
				}
			}
		],
		"tags": [
			"prod",
			"london"
		],
		"capacity_stats": {
			"total_capacity_bytes": 140226224128,
			"available_capacity_bytes": 92735586304,
			"provisioned_capacity_bytes": 0
		},
		"driver_capacity_stats": null,
		"labels": null
	}`
	fakeRT := &FakeRoundTripper{message: body, status: http.StatusOK}
	client := newTestClient(fakeRT)
	pool, err := client.PoolCreate(
		types.PoolCreateOptions{
			Name:            "unit01",
			Description:     "Unit test pool",
			Default:         false,
			DefaultDriver:   "xyz",
			ControllerNames: []string{"controller_a", "controller_b", "controller_c"},
			DriverNames:     []string{"driver_x", "driver_y", "driver_z"},
			Active:          true,
			Labels: map[string]string{
				"foo": "bar",
			},
			Context: context.Background(),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if pool == nil {
		t.Fatalf("PoolCreate: Wrong return value. Wanted pool. Got %v.", pool)
	}
	if len(pool.ID) != 36 {
		t.Errorf("PoolCreate: Wrong return value. Wanted 34 character UUID. Got %d. (%s)", len(pool.ID), pool.ID)
	}
	req := fakeRT.requests[0]
	expectedMethod := "POST"
	if req.Method != expectedMethod {
		t.Errorf("PoolCreate(): Wrong HTTP method. Want %s. Got %s.", expectedMethod, req.Method)
	}
	u, _ := url.Parse(client.getAPIPath(PoolAPIPrefix, url.Values{}, false))
	if req.URL.Path != u.Path {
		t.Errorf("PoolCreate(): Wrong request path. Want %q. Got %q.", u.Path, req.URL.Path)
	}
}

func TestPool(t *testing.T) {
	body := `{
                "active": true,
                "capacity_stats": {
                    "available_capacity_bytes": 80296787968,
                    "provisioned_capacity_bytes": 5368709120,
                    "total_capacity_bytes": 103440351232
                },
                "config": {
                    "data_dir": "/var/lib/storageos/data"
                },
                "controller_name": "storageos-1",
                "datacentre": "",
                "description": "Default storage pool",
                "driver_name": "filesystem",
                "id": "2935b1b9-a8af-121c-9e79-a64c637f0ee9",
                "name": "default",
                "pool_id": "b4c87d6c-2958-6283-128b-f767153938ad",
                "tags": [
                    "prod",
                    "london"
                ],
                "tenant": "",
                "type": ""
            }`
	var expected types.Pool
	if err := json.Unmarshal([]byte(body), &expected); err != nil {
		t.Fatal(err)
	}
	fakeRT := &FakeRoundTripper{message: body, status: http.StatusOK}
	client := newTestClient(fakeRT)
	name := "default"
	pool, err := client.Pool(name)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(pool, &expected) {
		t.Errorf("Pool: Wrong return value. Want %#v. Got %#v.", expected, pool)
	}
	req := fakeRT.requests[0]
	expectedMethod := "GET"
	if req.Method != expectedMethod {
		t.Errorf("InspectPool(%q): Wrong HTTP method. Want %s. Got %s.", name, expectedMethod, req.Method)
	}
	u, _ := url.Parse(client.getAPIPath(PoolAPIPrefix+"/"+name, url.Values{}, false))
	if req.URL.Path != u.Path {
		t.Errorf("PoolCreate(%q): Wrong request path. Want %q. Got %q.", name, u.Path, req.URL.Path)
	}
}

func TestPoolDelete(t *testing.T) {
	name := "testdelete"
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusNoContent}
	client := newTestClient(fakeRT)
	if err := client.PoolDelete(types.DeleteOptions{Name: name}); err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	expectedMethod := "DELETE"
	if req.Method != expectedMethod {
		t.Errorf("PoolDelete(%q): Wrong HTTP method. Want %s. Got %s.", name, expectedMethod, req.Method)
	}
	u, _ := url.Parse(client.getAPIPath(PoolAPIPrefix+"/"+name, url.Values{}, false))
	if req.URL.Path != u.Path {
		t.Errorf("PoolDelete(%q): Wrong request path. Want %q. Got %q.", name, u.Path, req.URL.Path)
	}
}

func TestPoolDeleteNotFound(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "no such pool", status: http.StatusNotFound})
	if err := client.PoolDelete(types.DeleteOptions{Name: "testdeletenotfound"}); err != ErrNoSuchPool {
		t.Errorf("PoolDelete: wrong error. Want %#v. Got %#v.", ErrNoSuchPool, err)
	}
}

func TestPoolDeleteInUse(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "pool in use and cannot be removed", status: http.StatusConflict})
	if err := client.PoolDelete(types.DeleteOptions{Name: "testdeletinuse"}); err != ErrPoolInUse {
		t.Errorf("PoolDelete: wrong error. Want %#v. Got %#v.", ErrNamespaceInUse, err)
	}
}

func TestPoolDeleteForce(t *testing.T) {
	name := "testdelete"
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusNoContent}
	client := newTestClient(fakeRT)
	if err := client.PoolDelete(types.DeleteOptions{Name: name, Force: true}); err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	vals := req.URL.Query()
	if len(vals) == 0 {
		t.Error("PoolDelete: query string empty. Expected force=1.")
	}
	force := vals.Get("force")
	if force != "1" {
		t.Errorf("PoolDelete(%q): Force not set. Want %q. Got %q.", name, "1", force)
	}
}
