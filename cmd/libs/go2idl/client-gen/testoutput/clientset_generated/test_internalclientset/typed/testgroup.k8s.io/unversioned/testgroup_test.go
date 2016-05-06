/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"net/http"
	"net/url"
	"testing"

	testgroup "k8s.io/kubernetes/cmd/libs/go2idl/client-gen/test_apis/testgroup.k8s.io"
	_ "k8s.io/kubernetes/cmd/libs/go2idl/client-gen/test_apis/testgroup.k8s.io/install"
	. "k8s.io/kubernetes/cmd/libs/go2idl/client-gen/testoutput/clientset_generated/test_internalclientset/typed/testgroup.k8s.io/unversioned"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"
	"k8s.io/kubernetes/pkg/labels"
)

var testHelper testapi.TestGroup

func init() {
	if _, found := testapi.Groups[testgroup.SchemeGroupVersion.Group]; found {
		return
	}
	testapi.Groups[testgroup.SchemeGroupVersion.Group] = testapi.NewTestGroup(
		registered.GroupOrDie(testgroup.SchemeGroupVersion.Group).GroupVersion,
		testgroup.SchemeGroupVersion,
		api.Scheme.KnownTypes(testgroup.SchemeGroupVersion))
	testHelper = testapi.Groups[testgroup.SchemeGroupVersion.Group]
}

type DecoratedSimpleClient struct {
	*TestgroupClient
	simpleClient simple.Client
}

func (c *DecoratedSimpleClient) Setup(t *testing.T) *DecoratedSimpleClient {
	c.simpleClient.Setup(t)
	url := c.simpleClient.ServerURL()
	c.TestgroupClient = NewForConfigOrDie(&restclient.Config{
		Host: url,
	})
	return c
}

func TestCreateTestTypes(t *testing.T) {
	ns := api.NamespaceDefault
	requestTestType := &testgroup.TestType{}
	c := DecoratedSimpleClient{
		simpleClient: simple.Client{
			Request: simple.Request{Method: "POST", Path: testHelper.ResourcePath("testtypes", ns, ""), Query: simple.BuildQueryValues(nil), Body: requestTestType},
			Response: simple.Response{
				StatusCode: http.StatusOK,
				Body:       requestTestType,
			},
		},
	}
	receivedTestType, err := c.Setup(t).TestTypes(ns).Create(requestTestType)
	c.simpleClient.Validate(t, receivedTestType, err)
}

func TestUpdateTestType(t *testing.T) {
	ns := api.NamespaceDefault
	requestTestType := &testgroup.TestType{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			ResourceVersion: "1",
			Labels: map[string]string{
				"foo":  "bar",
				"name": "baz",
			},
		},
	}
	c := DecoratedSimpleClient{
		simpleClient: simple.Client{
			Request:  simple.Request{Method: "PUT", Path: testHelper.ResourcePath("testtypes", ns, "foo"), Query: simple.BuildQueryValues(nil)},
			Response: simple.Response{StatusCode: http.StatusOK, Body: requestTestType},
		},
	}
	receivedTestType, err := c.Setup(t).TestTypes(ns).Update(requestTestType)
	c.simpleClient.Validate(t, receivedTestType, err)
}

func TestUpdateStatusTestType(t *testing.T) {
	ns := api.NamespaceDefault
	requestTestType := &testgroup.TestType{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			ResourceVersion: "1",
			Labels: map[string]string{
				"foo":  "bar",
				"name": "baz",
			},
		},
		Status: testgroup.TestTypeStatus{Blah: "I'm in good status"},
	}
	c := DecoratedSimpleClient{
		simpleClient: simple.Client{
			Request:  simple.Request{Method: "PUT", Path: testHelper.ResourcePath("testtypes", ns, "foo"), Query: simple.BuildQueryValues(nil)},
			Response: simple.Response{StatusCode: http.StatusOK, Body: requestTestType},
		},
	}
	receivedTestType, err := c.Setup(t).TestTypes(ns).Update(requestTestType)
	c.simpleClient.Validate(t, receivedTestType, err)
}

func TestDeleteTestType(t *testing.T) {
	ns := api.NamespaceDefault
	c := DecoratedSimpleClient{
		simpleClient: simple.Client{
			Request:  simple.Request{Method: "DELETE", Path: testHelper.ResourcePath("testtypes", ns, "foo"), Query: simple.BuildQueryValues(nil)},
			Response: simple.Response{StatusCode: http.StatusOK},
		},
	}
	err := c.Setup(t).TestTypes(ns).Delete("foo", nil)
	c.simpleClient.Validate(t, nil, err)
}

func TestGetTestType(t *testing.T) {
	ns := api.NamespaceDefault
	c := DecoratedSimpleClient{
		simpleClient: simple.Client{
			Request: simple.Request{Method: "GET", Path: testHelper.ResourcePath("testtypes", ns, "foo"), Query: simple.BuildQueryValues(nil)},
			Response: simple.Response{
				StatusCode: http.StatusOK,
				Body: &testgroup.TestType{
					ObjectMeta: api.ObjectMeta{
						Labels: map[string]string{
							"foo":  "bar",
							"name": "baz",
						},
					},
				},
			},
		},
	}
	receivedTestType, err := c.Setup(t).TestTypes(ns).Get("foo")
	c.simpleClient.Validate(t, receivedTestType, err)
}

func TestGetTestTypeWithNoName(t *testing.T) {
	ns := api.NamespaceDefault
	c := DecoratedSimpleClient{
		simpleClient: simple.Client{Error: true},
	}
	receivedTestType, err := c.Setup(t).TestTypes(ns).Get("")
	if (err != nil) && (err.Error() != simple.NameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", simple.NameRequiredError, err)
	}

	c.simpleClient.Validate(t, receivedTestType, err)
}

func TestListEmptyTestTypes(t *testing.T) {
	ns := api.NamespaceDefault
	c := DecoratedSimpleClient{
		simpleClient: simple.Client{
			Request:  simple.Request{Method: "GET", Path: testHelper.ResourcePath("testtypes", ns, ""), Query: simple.BuildQueryValues(nil)},
			Response: simple.Response{StatusCode: http.StatusOK, Body: &testgroup.TestTypeList{}},
		},
	}
	podList, err := c.Setup(t).TestTypes(ns).List(api.ListOptions{})
	c.simpleClient.Validate(t, podList, err)
}

func TestListTestTypes(t *testing.T) {
	ns := api.NamespaceDefault
	c := DecoratedSimpleClient{
		simpleClient: simple.Client{
			Request: simple.Request{Method: "GET", Path: testHelper.ResourcePath("testtypes", ns, ""), Query: simple.BuildQueryValues(nil)},
			Response: simple.Response{StatusCode: http.StatusOK,
				Body: &testgroup.TestTypeList{
					Items: []testgroup.TestType{
						{
							ObjectMeta: api.ObjectMeta{
								Labels: map[string]string{
									"foo":  "bar",
									"name": "baz",
								},
							},
						},
					},
				},
			},
		},
	}
	receivedTestTypeList, err := c.Setup(t).TestTypes(ns).List(api.ListOptions{})
	c.simpleClient.Validate(t, receivedTestTypeList, err)
}

func TestListTestTypesLabels(t *testing.T) {
	ns := api.NamespaceDefault
	labelSelectorQueryParamName := unversioned.LabelSelectorQueryParam(testHelper.GroupVersion().String())
	c := DecoratedSimpleClient{
		simpleClient: simple.Client{
			Request: simple.Request{
				Method: "GET",
				Path:   testHelper.ResourcePath("testtypes", ns, ""),
				Query:  simple.BuildQueryValues(url.Values{labelSelectorQueryParamName: []string{"foo=bar,name=baz"}})},
			Response: simple.Response{
				StatusCode: http.StatusOK,
				Body: &testgroup.TestTypeList{
					Items: []testgroup.TestType{
						{
							ObjectMeta: api.ObjectMeta{
								Labels: map[string]string{
									"foo":  "bar",
									"name": "baz",
								},
							},
						},
					},
				},
			},
		},
	}
	c.Setup(t)
	c.simpleClient.QueryValidator[labelSelectorQueryParamName] = simple.ValidateLabels
	selector := labels.Set{"foo": "bar", "name": "baz"}.AsSelector()
	options := api.ListOptions{LabelSelector: selector}
	receivedTestTypeList, err := c.TestTypes(ns).List(options)
	c.simpleClient.Validate(t, receivedTestTypeList, err)
}

func TestExpansionInterface(t *testing.T) {
	c := New(nil)
	if e, a := "hello!", c.TestTypes("").Hello(); e != a {
		t.Errorf("expansion failed")
	}
}
