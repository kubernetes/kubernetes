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

func TestVolumeList(t *testing.T) {
	volumesData := `[{
    "id": "6b7afe82-565f-a627-4696-22457da5da9c",
    "master": {
        "controller": "",
        "id": "",
        "inode": 0,
        "status": "",
        "health": "",
        "created_at": "0001-01-01T00:00:00Z"
    },
    "replicas": null,
    "created_by": "storageos",
    "name": "test02",
    "status": "pending",
    "status_message": "",
    "health": "",
    "pool": "213498fb-ead9-2a48-92e6-4dac2020f2ed",
    "description": "",
    "size": 10,
    "inode": 0,
    "volume_groups": [],
    "tags": ["filesystem"],
    "mounted": false,
    "no_of_mounts": 0,
    "mounted_by": "",
    "mounted_at": "0001-01-01T00:00:00Z",
    "created_at": "0001-01-01T00:00:00Z"
}, {
    "id": "ef897b9f-0b47-08ee-b669-0a2057df981c",
    "master": {
        "controller": "b3eb8d63-4f1b-9ef5-a504-7d02d604feb4",
        "id": "55fb06cb-263d-08bf-584e-e5b889166f3b",
        "inode": 41560,
        "status": "active",
        "health": "",
        "created_at": "2017-01-25T02:17:05.507557244Z"
    },
    "replicas": null,
    "created_by": "storageos",
    "name": "test01",
    "status": "active",
    "status_message": "",
    "health": "",
    "pool": "213498fb-ead9-2a48-92e6-4dac2020f2ed",
    "description": "",
    "size": 10,
    "inode": 41397,
    "volume_groups": null,
    "tags": ["filesystem", "compression"],
    "mounted": false,
    "no_of_mounts": 0,
    "mounted_by": "",
    "mounted_at": "0001-01-01T00:00:00Z",
    "created_at": "0001-01-01T00:00:00Z"
}]`

	var expected []*types.Volume
	if err := json.Unmarshal([]byte(volumesData), &expected); err != nil {
		t.Fatal(err)
	}

	client := newTestClient(&FakeRoundTripper{message: volumesData, status: http.StatusOK})
	volumes, err := client.VolumeList(types.ListOptions{Namespace: "projA"})
	if err != nil {
		t.Error(err)
	}
	if !reflect.DeepEqual(volumes, expected) {
		t.Errorf("Volumes: Wrong return value. Want %#v. Got %#v.", expected, volumes)
	}
}

func TestVolumeListLabelSelector(t *testing.T) {

	fakeRT := &FakeRoundTripper{message: `[]`, status: http.StatusOK}
	client := newTestClient(fakeRT)
	_, err := client.VolumeList(types.ListOptions{LabelSelector: "env=prod"})
	if err != nil {
		t.Error(err)
	}

	req := fakeRT.requests[0]
	expectedVals := url.Values{}
	expectedVals.Add("labelSelector", "env=prod")
	u, _ := url.Parse(client.getAPIPath(VolumeAPIPrefix, expectedVals, false))
	if req.URL.Path != u.Path {
		t.Errorf("TestVolumeListLabelSelector(): Wrong request path. Want %q. Got %q.", u.Path, req.URL.Path)
	}
}

func TestVolumeCreate(t *testing.T) {
	body := `{
				"created_at": "0001-01-01T00:00:00Z",
				"created_by": "storageos",
				"datacentre": "",
				"description": "Kubernetes volume",
				"health": "",
				"id": "671e90a2-06f9-9cd7-b4ee-d1338dfe31ee",
				"inode": 137190,
				"labels": {
						"storageos.driver": "filesystem"
				},
				"master": {
						"controller": "01c43d34-89f8-83d3-422b-43536a0f25e6",
						"created_at": "2017-02-15T01:40:44.792120679Z",
						"health": "",
						"id": "2a611b3f-8e23-eaa3-537a-d0b635bdd6a5",
						"inode": 166017,
						"status": "active"
				},
				"mounted": false,
				"mounted_at": "0001-01-01T00:00:00Z",
				"mounted_by": "",
				"name": "pvc-c46b39a3-f31f-11e6-9fe1-08002736b526",
				"no_of_mounts": 0,
				"pool": "b4c87d6c-2958-6283-128b-f767153938ad",
				"replicas": [],
				"size": 5,
				"status": "active",
				"status_message": "volume was affected by controller with ID 01c43d34-89f8-83d3-422b-43536a0f25e6 submodule health changes",
				"tenant": ""
		}`
	fakeRT := &FakeRoundTripper{message: body, status: http.StatusOK}
	client := newTestClient(fakeRT)
	namespace := "projA"
	volume, err := client.VolumeCreate(
		types.VolumeCreateOptions{
			Name:        "unit01",
			Description: "Unit test volume",
			Pool:        "default",
			Namespace:   namespace,
			Labels: map[string]string{
				"foo": "bar",
			},
			Context: context.Background(),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if volume == nil {
		t.Fatalf("VolumeCreate(): Wrong return value. Wanted volume. Got %v.", volume)
	}
	if len(volume.ID) != 36 {
		t.Errorf("VolumeCreate(): Wrong return value. Wanted 34 character UUID. Got %d. (%s)", len(volume.ID), volume.ID)
	}
	req := fakeRT.requests[0]
	expectedMethod := "POST"
	if req.Method != expectedMethod {
		t.Errorf("VolumeCreate(): Wrong HTTP method. Want %s. Got %s.", expectedMethod, req.Method)
	}
	path, _ := namespacedPath(namespace, VolumeAPIPrefix)
	u, _ := url.Parse(client.getAPIPath(path, url.Values{}, false))
	if req.URL.Path != u.Path {
		t.Errorf("VolumeCreate(): Wrong request path. Want %q. Got %q.", u.Path, req.URL.Path)
	}
}

func TestVolume(t *testing.T) {
	body := `{
        "created_at": "0001-01-01T00:00:00Z",
        "created_by": "storageos",
        "datacentre": "",
        "description": "Kubernetes volume",
        "health": "",
        "id": "671e90a2-06f9-9cd7-b4ee-d1338dfe31ee",
        "inode": 137190,
        "labels": {
            "storageos.driver": "filesystem"
        },
        "master": {
            "controller": "01c43d34-89f8-83d3-422b-43536a0f25e6",
            "created_at": "2017-02-15T01:40:44.792120679Z",
            "health": "",
            "id": "2a611b3f-8e23-eaa3-537a-d0b635bdd6a5",
            "inode": 166017,
            "status": "active"
        },
        "mounted": false,
        "mounted_at": "0001-01-01T00:00:00Z",
        "mounted_by": "",
        "name": "pvc-c46b39a3-f31f-11e6-9fe1-08002736b526",
        "no_of_mounts": 0,
        "pool": "b4c87d6c-2958-6283-128b-f767153938ad",
        "replicas": [],
        "size": 5,
        "status": "active",
        "status_message": "volume was affected by controller with ID 01c43d34-89f8-83d3-422b-43536a0f25e6 submodule health changes",
        "tenant": ""
    }`
	var expected types.Volume
	if err := json.Unmarshal([]byte(body), &expected); err != nil {
		t.Fatal(err)
	}
	fakeRT := &FakeRoundTripper{message: body, status: http.StatusOK}
	client := newTestClient(fakeRT)
	name := "tardis"
	namespace := "projA"
	volume, err := client.Volume(namespace, name)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(volume, &expected) {
		t.Errorf("Volume: Wrong return value. Want %#v. Got %#v.", expected, volume)
	}
	req := fakeRT.requests[0]
	expectedMethod := "GET"
	if req.Method != expectedMethod {
		t.Errorf("InspectVolume(%q): Wrong HTTP method. Want %s. Got %s.", name, expectedMethod, req.Method)
	}
	path, _ := namespacedRefPath(namespace, VolumeAPIPrefix, name)
	u, _ := url.Parse(client.getAPIPath(path, url.Values{}, false))
	if req.URL.Path != u.Path {
		t.Errorf("VolumeCreate(%q): Wrong request path. Want %q. Got %q.", name, u.Path, req.URL.Path)
	}
}

func TestVolumeDelete(t *testing.T) {
	name := "test"
	namespace := "projA"
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusNoContent}
	client := newTestClient(fakeRT)
	err := client.VolumeDelete(
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
		t.Errorf("VolumeDelete(%q): Wrong HTTP method. Want %s. Got %s.", name, expectedMethod, req.Method)
	}
	path, _ := namespacedRefPath(namespace, VolumeAPIPrefix, name)
	u, _ := url.Parse(client.getAPIPath(path, url.Values{}, false))
	if req.URL.Path != u.Path {
		t.Errorf("VolumeDelete(%q): Wrong request path. Want %q. Got %q.", name, u.Path, req.URL.Path)
	}
}

func TestVolumeDeleteNotFound(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "no such volume", status: http.StatusNotFound})
	err := client.VolumeDelete(
		types.DeleteOptions{
			Name:      "badname",
			Namespace: "badnamespace",
		},
	)
	if err != ErrNoSuchVolume {
		t.Errorf("VolumeDelete(%q): wrong error. Want %#v. Got %#v.", "badname", ErrNoSuchVolume, err)
	}
}

func TestVolumeDeleteInUse(t *testing.T) {
	name := "test"
	namespace := "projA"
	client := newTestClient(&FakeRoundTripper{message: "volume in use and cannot be removed", status: http.StatusConflict})
	err := client.VolumeDelete(
		types.DeleteOptions{
			Name:      name,
			Namespace: namespace,
		},
	)
	if err != ErrVolumeInUse {
		t.Errorf("VolumeDelete(%q): wrong error. Want %#v. Got %#v.", name, ErrVolumeInUse, err)
	}
}

func TestVolumeDeleteForce(t *testing.T) {
	name := "testdelete"
	namespace := "projA"
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusNoContent}
	client := newTestClient(fakeRT)
	if err := client.VolumeDelete(types.DeleteOptions{Name: name, Namespace: namespace, Force: true}); err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	vals := req.URL.Query()
	if len(vals) == 0 {
		t.Error("VolumeDelete: query string empty. Expected force=1.")
	}
	force := vals.Get("force")
	if force != "1" {
		t.Errorf("VolumeDelete(%q): Force not set. Want %q. Got %q.", name, "1", force)
	}
}

func TestVolumeMount(t *testing.T) {
	name := "test"
	namespace := "projA"
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusNoContent}
	client := newTestClient(fakeRT)
	err := client.VolumeMount(
		types.VolumeMountOptions{
			Name:      name,
			Namespace: namespace,
			Client:    "clientA",
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	expectedMethod := "POST"
	if req.Method != expectedMethod {
		t.Errorf("VolumeMount(%q): Wrong HTTP method. Want %s. Got %s.", name, expectedMethod, req.Method)
	}
	path, _ := namespacedRefPath(namespace, VolumeAPIPrefix, name)
	u, _ := url.Parse(client.getAPIPath(path+"/mount", url.Values{}, false))
	if req.URL.Path != u.Path {
		t.Errorf("VolumeMount(%q): Wrong request path. Want %q. Got %q.", name, u.Path, req.URL.Path)
	}
}

func TestVolumeUnmount(t *testing.T) {
	name := "test"
	namespace := "projA"
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusNoContent}
	client := newTestClient(fakeRT)
	err := client.VolumeUnmount(
		types.VolumeUnmountOptions{
			Name:      name,
			Namespace: namespace,
			Client:    "clientA",
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	expectedMethod := "POST"
	if req.Method != expectedMethod {
		t.Errorf("VolumeUnmount(%q): Wrong HTTP method. Want %s. Got %s.", name, expectedMethod, req.Method)
	}
	path, _ := namespacedRefPath(namespace, VolumeAPIPrefix, name)
	u, _ := url.Parse(client.getAPIPath(path+"/unmount", url.Values{}, false))
	if req.URL.Path != u.Path {
		t.Errorf("VolumeUnount(%q): Wrong request path. Want %q. Got %q.", name, u.Path, req.URL.Path)
	}
}
