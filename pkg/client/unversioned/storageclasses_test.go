/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package unversioned_test

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"
)

func getStorageClassResourceName() string {
	return "storageclasses"
}

func TestListStorageClasses(t *testing.T) {
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Storage.ResourcePath(getStorageClassResourceName(), "", ""),
		},
		Response: simple.Response{StatusCode: 200,
			Body: &storage.StorageClassList{
				Items: []storage.StorageClass{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "foo",
							Labels: map[string]string{
								"foo":  "bar",
								"name": "baz",
							},
						},
						Provisioner: "aaa",
					},
				},
			},
		},
	}
	receivedSCList, err := c.Setup(t).Storage().StorageClasses().List(api.ListOptions{})
	c.Validate(t, receivedSCList, err)
}

func TestGetStorageClass(t *testing.T) {
	c := &simple.Client{
		Request: simple.Request{Method: "GET", Path: testapi.Storage.ResourcePath(getStorageClassResourceName(), "", "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{
			StatusCode: 200,
			Body: &storage.StorageClass{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Provisioner: "aaa",
			},
		},
	}
	receivedSC, err := c.Setup(t).Storage().StorageClasses().Get("foo")
	c.Validate(t, receivedSC, err)
}

func TestGetStorageClassWithNoName(t *testing.T) {
	c := &simple.Client{Error: true}
	receivedSC, err := c.Setup(t).Storage().StorageClasses().Get("")
	if (err != nil) && (err.Error() != simple.NameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", simple.NameRequiredError, err)
	}

	c.Validate(t, receivedSC, err)
}

func TestUpdateStorageClass(t *testing.T) {
	requestSC := &storage.StorageClass{
		ObjectMeta:  api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
		Provisioner: "aaa",
	}
	c := &simple.Client{
		Request: simple.Request{Method: "PUT", Path: testapi.Storage.ResourcePath(getStorageClassResourceName(), "", "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{
			StatusCode: 200,
			Body: &storage.StorageClass{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Provisioner: "aaa",
			},
		},
	}
	receivedSC, err := c.Setup(t).Storage().StorageClasses().Update(requestSC)
	c.Validate(t, receivedSC, err)
}

func TestDeleteStorageClass(t *testing.T) {
	c := &simple.Client{
		Request:  simple.Request{Method: "DELETE", Path: testapi.Storage.ResourcePath(getStorageClassResourceName(), "", "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200},
	}
	err := c.Setup(t).Storage().StorageClasses().Delete("foo")
	c.Validate(t, nil, err)
}

func TestCreateStorageClass(t *testing.T) {
	requestSC := &storage.StorageClass{
		ObjectMeta:  api.ObjectMeta{Name: "foo"},
		Provisioner: "aaa",
	}
	c := &simple.Client{
		Request: simple.Request{Method: "POST", Path: testapi.Storage.ResourcePath(getStorageClassResourceName(), "", ""), Body: requestSC, Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{
			StatusCode: 200,
			Body: &storage.StorageClass{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Provisioner: "aaa",
			},
		},
	}
	receivedSC, err := c.Setup(t).Storage().StorageClasses().Create(requestSC)
	c.Validate(t, receivedSC, err)
}
