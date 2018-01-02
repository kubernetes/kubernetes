package storageos

import (
	"encoding/json"
	"net/http"
	"net/url"
	"reflect"
	"testing"

	"github.com/storageos/go-api/types"
)

func TestControllerList(t *testing.T) {
	controllersData := `[
    {
        "address": "172.28.128.121",
        "apiPort": 0,
        "capacityStats": {
            "availableCapacityBytes": 13057638400,
            "provisionedCapacityBytes": 0,
            "totalCapacityBytes": 18392936448
        },
        "controllerGroups": null,
        "description": "",
        "dfsPort": 0,
        "health": "healthy",
        "healthUpdatedAt": "2017-04-14T16:57:09.878839027Z",
        "hostID": 61111,
        "id": "c7981fe8-824e-50cb-e858-77c0a8b67f68",
        "labels": null,
        "name": "storageos-3-26316",
        "natsClusterPort": 0,
        "natsPort": 0,
        "poolStats": {
            "b736e39d-3512-b7a1-b7db-74ebebece898": {
                "filesystem": {
                    "availableCapacityBytes": 13057638400,
                    "provisionedCapacityBytes": 0,
                    "totalCapacityBytes": 18392936448
                }
            }
        },
        "scheduler": true,
        "serfPort": 0,
        "tags": null,
        "version": "StorageOS 912d0be, Built: 2017-04-14T165029Z",
        "versionInfo": {
            "storageos": {
                "apiVersion": "1",
                "arch": "amd64",
                "buildDate": "2017-04-14T165029Z",
                "experimental": false,
                "goVersion": "go1.7.3",
                "kernelVersion": "",
                "name": "storageos",
                "os": "linux",
                "revision": "912d0be",
                "version": "912d0be"
            }
        },
        "volumeStats": {
            "masterVolumeCount": 3,
            "replicaVolumeCount": 0,
            "virtualVolumeCount": 0
        }
    },
    {
        "address": "172.28.128.109",
        "apiPort": 0,
        "capacityStats": {
            "availableCapacityBytes": 8004571136,
            "provisionedCapacityBytes": 0,
            "totalCapacityBytes": 18392936448
        },
        "controllerGroups": null,
        "description": "",
        "dfsPort": 0,
        "health": "healthy",
        "healthUpdatedAt": "2017-04-16T19:15:16.832182995Z",
        "hostID": 46262,
        "id": "000e00db-8b6c-a6d3-e635-865bec82b1d9",
        "labels": null,
        "name": "storageos-1-26316",
        "natsClusterPort": 0,
        "natsPort": 0,
        "poolStats": {
            "b736e39d-3512-b7a1-b7db-74ebebece898": {
                "filesystem": {
                    "availableCapacityBytes": 8004571136,
                    "provisionedCapacityBytes": 0,
                    "totalCapacityBytes": 18392936448
                }
            }
        },
        "scheduler": false,
        "serfPort": 0,
        "tags": null,
        "version": "StorageOS b38bf97, Built: 2017-04-16T191439Z",
        "versionInfo": {
            "storageos": {
                "apiVersion": "1",
                "arch": "amd64",
                "buildDate": "2017-04-16T191439Z",
                "experimental": false,
                "goVersion": "go1.7.3",
                "kernelVersion": "",
                "name": "storageos",
                "os": "linux",
                "revision": "b38bf97",
                "version": "b38bf97"
            }
        },
        "volumeStats": {
            "masterVolumeCount": 0,
            "replicaVolumeCount": 0,
            "virtualVolumeCount": 0
        }
    },
    {
        "address": "172.28.128.115",
        "apiPort": 0,
        "capacityStats": {
            "availableCapacityBytes": 12749434880,
            "provisionedCapacityBytes": 0,
            "totalCapacityBytes": 18392936448
        },
        "controllerGroups": null,
        "description": "",
        "dfsPort": 0,
        "health": "healthy",
        "healthUpdatedAt": "2017-04-17T09:13:43.300203524Z",
        "hostID": 30689,
        "id": "70e9ea5f-be0c-9f82-391a-41d25538060b",
        "labels": null,
        "name": "storageos-2-26316",
        "natsClusterPort": 0,
        "natsPort": 0,
        "poolStats": {
            "b736e39d-3512-b7a1-b7db-74ebebece898": {
                "filesystem": {
                    "availableCapacityBytes": 12749434880,
                    "provisionedCapacityBytes": 0,
                    "totalCapacityBytes": 18392936448
                }
            }
        },
        "scheduler": false,
        "serfPort": 0,
        "tags": null,
        "version": "StorageOS 912d0be, Built: 2017-04-14T165029Z",
        "versionInfo": {
            "storageos": {
                "apiVersion": "1",
                "arch": "amd64",
                "buildDate": "2017-04-14T165029Z",
                "experimental": false,
                "goVersion": "go1.7.3",
                "kernelVersion": "",
                "name": "storageos",
                "os": "linux",
                "revision": "912d0be",
                "version": "912d0be"
            }
        },
        "volumeStats": {
            "masterVolumeCount": 0,
            "replicaVolumeCount": 1,
            "virtualVolumeCount": 0
        }
    }
]`

	var expected []*types.Controller
	if err := json.Unmarshal([]byte(controllersData), &expected); err != nil {
		t.Fatal(err)
	}

	client := newTestClient(&FakeRoundTripper{message: controllersData, status: http.StatusOK})
	controllers, err := client.ControllerList(types.ListOptions{Namespace: "projA"})
	if err != nil {
		t.Error(err)
	}
	if !reflect.DeepEqual(controllers, expected) {
		t.Errorf("Controllers: Wrong return value. Want %#v. Got %#v.", expected, controllers)
	}
}

func TestControllerListLabelSelector(t *testing.T) {

	fakeRT := &FakeRoundTripper{message: `[]`, status: http.StatusOK}
	client := newTestClient(fakeRT)
	_, err := client.ControllerList(types.ListOptions{LabelSelector: "env=prod"})
	if err != nil {
		t.Error(err)
	}

	req := fakeRT.requests[0]
	expectedVals := url.Values{}
	expectedVals.Add("labelSelector", "env=prod")
	u, _ := url.Parse(client.getAPIPath(ControllerAPIPrefix, expectedVals, false))
	if req.URL.Path != u.Path {
		t.Errorf("TestControllerListLabelSelector(): Wrong request path. Want %q. Got %q.", u.Path, req.URL.Path)
	}
}

func TestController(t *testing.T) {
	body := `{
    "address": "172.28.128.121",
    "apiPort": 0,
    "capacityStats": {
        "availableCapacityBytes": 13057564672,
        "provisionedCapacityBytes": 0,
        "totalCapacityBytes": 18392936448
    },
    "controllerGroups": null,
    "description": "",
    "dfsPort": 0,
    "health": "healthy",
    "healthUpdatedAt": "2017-04-14T16:57:09.878839027Z",
    "hostID": 61111,
    "id": "c7981fe8-824e-50cb-e858-77c0a8b67f68",
    "labels": null,
    "name": "storageos-3-26316",
    "natsClusterPort": 0,
    "natsPort": 0,
    "poolStats": {
        "b736e39d-3512-b7a1-b7db-74ebebece898": {
            "filesystem": {
                "availableCapacityBytes": 13057564672,
                "provisionedCapacityBytes": 0,
                "totalCapacityBytes": 18392936448
            }
        }
    },
    "scheduler": true,
    "serfPort": 0,
    "tags": null,
    "version": "StorageOS 912d0be, Built: 2017-04-14T165029Z",
    "versionInfo": {
        "storageos": {
            "apiVersion": "1",
            "arch": "amd64",
            "buildDate": "2017-04-14T165029Z",
            "experimental": false,
            "goVersion": "go1.7.3",
            "kernelVersion": "",
            "name": "storageos",
            "os": "linux",
            "revision": "912d0be",
            "version": "912d0be"
        }
    },
    "volumeStats": {
        "masterVolumeCount": 3,
        "replicaVolumeCount": 0,
        "virtualVolumeCount": 0
    }
}`
	var expected types.Controller
	if err := json.Unmarshal([]byte(body), &expected); err != nil {
		t.Fatal(err)
	}
	fakeRT := &FakeRoundTripper{message: body, status: http.StatusOK}
	client := newTestClient(fakeRT)
	name := "tardis"
	namespace := "projA"
	controller, err := client.Controller(name)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(controller, &expected) {
		t.Errorf("Controller: Wrong return value. Want %#v. Got %#v.", expected, controller)
	}
	req := fakeRT.requests[0]
	expectedMethod := "GET"
	if req.Method != expectedMethod {
		t.Errorf("InspectController(%q): Wrong HTTP method. Want %s. Got %s.", name, expectedMethod, req.Method)
	}
	path, _ := namespacedRefPath(namespace, ControllerAPIPrefix, name)
	u, _ := url.Parse(client.getAPIPath(path, url.Values{}, false))
	if req.URL.Path != u.Path {
		t.Errorf("ControllerCreate(%q): Wrong request path. Want %q. Got %q.", name, u.Path, req.URL.Path)
	}
}

func TestControllerDelete(t *testing.T) {
	name := "test"
	namespace := "projA"
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusNoContent}
	client := newTestClient(fakeRT)
	err := client.ControllerDelete(
		types.DeleteOptions{
			Name:      name,
			Namespace: namespace,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	expectedMethod := "DELETE"
	if req.Method != expectedMethod {
		t.Errorf("ControllerDelete(%q): Wrong HTTP method. Want %s. Got %s.", name, expectedMethod, req.Method)
	}
	path, _ := namespacedRefPath(namespace, ControllerAPIPrefix, name)
	u, _ := url.Parse(client.getAPIPath(path, url.Values{}, false))
	if req.URL.Path != u.Path {
		t.Errorf("ControllerDelete(%q): Wrong request path. Want %q. Got %q.", name, u.Path, req.URL.Path)
	}
}

func TestControllerDeleteNotFound(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "no such controller", status: http.StatusNotFound})
	err := client.ControllerDelete(
		types.DeleteOptions{
			Name:      "badname",
			Namespace: "badnamespace",
		},
	)
	if err != ErrNoSuchController {
		t.Errorf("ControllerDelete(%q): wrong error. Want %#v. Got %#v.", "badname", ErrNoSuchController, err)
	}
}

func TestControllerDeleteInUse(t *testing.T) {
	name := "test"
	namespace := "projA"
	client := newTestClient(&FakeRoundTripper{message: "controller in use and cannot be removed", status: http.StatusConflict})
	err := client.ControllerDelete(
		types.DeleteOptions{
			Name:      name,
			Namespace: namespace,
		},
	)
	if err != ErrControllerInUse {
		t.Errorf("ControllerDelete(%q): wrong error. Want %#v. Got %#v.", name, ErrControllerInUse, err)
	}
}

func TestControllerDeleteForce(t *testing.T) {
	name := "testdelete"
	namespace := "projA"
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusNoContent}
	client := newTestClient(fakeRT)
	if err := client.ControllerDelete(types.DeleteOptions{Name: name, Namespace: namespace, Force: true}); err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	vals := req.URL.Query()
	if len(vals) == 0 {
		t.Error("ControllerDelete: query string empty. Expected force=1.")
	}
	force := vals.Get("force")
	if force != "1" {
		t.Errorf("ControllerDelete(%q): Force not set. Want %q. Got %q.", name, "1", force)
	}
}
