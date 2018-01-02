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

func TestNamespaceList(t *testing.T) {
	namespacesData := `[
	{
		"id": "8768880e-8e2a-d064-f1f0-ed42a9222644",
		"name": "default",
		"displayName": "default",
		"description": "",
		"labels": null,
		"createdAt": "2017-02-17T20:04:47.031685185Z",
		"updatedAt": "2017-02-17T20:04:47.031685185Z"
	},
	{
		"id": "442d1b35-f851-fdf2-5e99-8d0e559ee7d7",
		"name": "simon",
		"displayName": "simon",
		"description": "",
		"labels": null,
		"createdAt": "2017-03-01T19:42:36.941317683Z",
		"updatedAt": "2017-03-01T19:42:36.941317683Z"
	}
]`

	var expected []*types.Namespace
	if err := json.Unmarshal([]byte(namespacesData), &expected); err != nil {
		t.Fatal(err)
	}

	client := newTestClient(&FakeRoundTripper{message: namespacesData, status: http.StatusOK})
	namespaces, err := client.NamespaceList(types.ListOptions{})
	if err != nil {
		t.Error(err)
	}
	if !reflect.DeepEqual(namespaces, expected) {
		t.Errorf("Namespaces: Wrong return value. Want %#v. Got %#v.", expected, namespaces)
	}
}

func TestNamespaceCreate(t *testing.T) {
	body := `	{
			"id": "442d1b35-f851-fdf2-5e99-8d0e559ee7d7",
			"name": "simon",
			"displayName": "simon",
			"description": "",
			"labels": null,
			"createdAt": "2017-03-01T19:42:36.941317683Z",
			"updatedAt": "2017-03-01T19:42:36.941317683Z"
		}`
	fakeRT := &FakeRoundTripper{message: body, status: http.StatusOK}
	client := newTestClient(fakeRT)
	namespace, err := client.NamespaceCreate(
		types.NamespaceCreateOptions{
			Name:        "nstest",
			DisplayName: "Namespace Test",
			Description: "Unit test namespace",
			Labels: map[string]string{
				"foo": "bar",
			},
			Context: context.Background(),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if namespace == nil {
		t.Fatalf("NamespaceCreate: Wrong return value. Wanted namespace. Got %v.", namespace)
	}
	if len(namespace.ID) != 36 {
		t.Errorf("NamespaceCreate: Wrong return value. Wanted 34 character UUID. Got %d. (%s)", len(namespace.ID), namespace.ID)
	}
	req := fakeRT.requests[0]
	expectedMethod := "POST"
	if req.Method != expectedMethod {
		t.Errorf("NamespaceCreate(): Wrong HTTP method. Want %s. Got %s.", expectedMethod, req.Method)
	}
	u, _ := url.Parse(client.getAPIPath(NamespaceAPIPrefix, url.Values{}, false))
	if req.URL.Path != u.Path {
		t.Errorf("NamespaceCreate(): Wrong request path. Want %q. Got %q.", u.Path, req.URL.Path)
	}
}

func TestNamespace(t *testing.T) {
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
                "description": "Default storage namespace",
                "driver_name": "filesystem",
                "id": "2935b1b9-a8af-121c-9e79-a64c637f0ee9",
                "name": "default",
                "namespace_id": "b4c87d6c-2958-6283-128b-f767153938ad",
                "tags": [
                    "prod",
                    "london"
                ],
                "tenant": "",
                "type": ""
            }`
	var expected types.Namespace
	if err := json.Unmarshal([]byte(body), &expected); err != nil {
		t.Fatal(err)
	}
	fakeRT := &FakeRoundTripper{message: body, status: http.StatusOK}
	client := newTestClient(fakeRT)
	name := "default"
	namespace, err := client.Namespace(name)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(namespace, &expected) {
		t.Errorf("Namespace: Wrong return value. Want %#v. Got %#v.", expected, namespace)
	}
	req := fakeRT.requests[0]
	expectedMethod := "GET"
	if req.Method != expectedMethod {
		t.Errorf("InspectNamespace(%q): Wrong HTTP method. Want %s. Got %s.", name, expectedMethod, req.Method)
	}
	u, _ := url.Parse(client.getAPIPath(NamespaceAPIPrefix+"/"+name, url.Values{}, false))
	if req.URL.Path != u.Path {
		t.Errorf("NamespaceCreate(%q): Wrong request path. Want %q. Got %q.", name, u.Path, req.URL.Path)
	}
}

func TestNamespaceDelete(t *testing.T) {
	name := "testdelete"
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusNoContent}
	client := newTestClient(fakeRT)
	if err := client.NamespaceDelete(types.DeleteOptions{Name: name}); err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	expectedMethod := "DELETE"
	if req.Method != expectedMethod {
		t.Errorf("NamespaceDelete(%q): Wrong HTTP method. Want %s. Got %s.", name, expectedMethod, req.Method)
	}
	u, _ := url.Parse(client.getAPIPath(NamespaceAPIPrefix+"/"+name, url.Values{}, false))
	if req.URL.Path != u.Path {
		t.Errorf("NamespaceDelete(%q): Wrong request path. Want %q. Got %q.", name, u.Path, req.URL.Path)
	}
}

func TestNamespaceDeleteNotFound(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "no such namespace", status: http.StatusNotFound})
	if err := client.NamespaceDelete(types.DeleteOptions{Name: "testdeletenotfound"}); err != ErrNoSuchNamespace {
		t.Errorf("TestNamespaceDeleteNotFound: wrong error. Want %#v. Got %#v.", ErrNoSuchNamespace, err)
	}
}

func TestNamespaceDeleteInUse(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "namespace in use and cannot be removed", status: http.StatusConflict})
	if err := client.NamespaceDelete(types.DeleteOptions{Name: "testdeleteinuseme"}); err != ErrNamespaceInUse {
		t.Errorf("TestNamespaceDeleteInUse: wrong error. Want %#v. Got %#v.", ErrNamespaceInUse, err)
	}
}

func TestNamespaceDeleteAnotherError(t *testing.T) {
	client := newTestClient(&FakeRoundTripper{message: "another error here", status: 410})
	if err := client.NamespaceDelete(types.DeleteOptions{Name: "testdeleteinuseme"}); err.Error() != "API error (410): another error here" {
		t.Errorf("TestNamespaceDeleteAnotherError: wrong error. Want %#v. Got %#v.", "another error here", err.Error())
	}
}

func TestNamespaceDeleteForce(t *testing.T) {
	name := "testdelete"
	fakeRT := &FakeRoundTripper{message: "", status: http.StatusNoContent}
	client := newTestClient(fakeRT)
	if err := client.NamespaceDelete(types.DeleteOptions{Name: name, Force: true}); err != nil {
		t.Fatal(err)
	}
	req := fakeRT.requests[0]
	vals := req.URL.Query()
	if len(vals) == 0 {
		t.Error("NamespaceDelete: query string empty. Expected force=1.")
	}
	force := vals.Get("force")
	if force != "1" {
		t.Errorf("NamespaceDelete(%q): Force not set. Want %q. Got %q.", name, "1", force)
	}
}
