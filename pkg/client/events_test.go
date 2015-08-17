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

package client

import (
	"net/url"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
)

func TestEventSearch(t *testing.T) {
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.ResourcePath("events", "baz", ""),
			Query: url.Values{
				api.FieldSelectorQueryParam(testapi.Version()): []string{
					getInvolvedObjectNameFieldLabel(testapi.Version()) + "=foo,",
					"involvedObject.namespace=baz,",
					"involvedObject.kind=Pod",
				},
				api.LabelSelectorQueryParam(testapi.Version()): []string{},
			},
		},
		Response: Response{StatusCode: 200, Body: &api.EventList{}},
	}
	eventList, err := c.Setup().Events("baz").Search(
		&api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:      "foo",
				Namespace: "baz",
				SelfLink:  testapi.SelfLink("pods", ""),
			},
		},
	)
	c.Validate(t, eventList, err)
}

func TestEventCreate(t *testing.T) {
	objReference := &api.ObjectReference{
		Kind:            "foo",
		Namespace:       "nm",
		Name:            "objref1",
		UID:             "uid",
		APIVersion:      "apiv1",
		ResourceVersion: "1",
	}
	timeStamp := util.Now()
	event := &api.Event{
		ObjectMeta: api.ObjectMeta{
			Namespace: api.NamespaceDefault,
		},
		InvolvedObject: *objReference,
		FirstTimestamp: timeStamp,
		LastTimestamp:  timeStamp,
		Count:          1,
	}
	c := &testClient{
		Request: testRequest{
			Method: "POST",
			Path:   testapi.ResourcePath("events", api.NamespaceDefault, ""),
			Body:   event,
		},
		Response: Response{StatusCode: 200, Body: event},
	}

	response, err := c.Setup().Events(api.NamespaceDefault).Create(event)

	if err != nil {
		t.Fatalf("%v should be nil.", err)
	}

	if e, a := *objReference, response.InvolvedObject; !reflect.DeepEqual(e, a) {
		t.Errorf("%#v != %#v.", e, a)
	}
}

func TestEventGet(t *testing.T) {
	objReference := &api.ObjectReference{
		Kind:            "foo",
		Namespace:       "nm",
		Name:            "objref1",
		UID:             "uid",
		APIVersion:      "apiv1",
		ResourceVersion: "1",
	}
	timeStamp := util.Now()
	event := &api.Event{
		ObjectMeta: api.ObjectMeta{
			Namespace: "other",
		},
		InvolvedObject: *objReference,
		FirstTimestamp: timeStamp,
		LastTimestamp:  timeStamp,
		Count:          1,
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.ResourcePath("events", "other", "1"),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: event},
	}

	response, err := c.Setup().Events("other").Get("1")

	if err != nil {
		t.Fatalf("%v should be nil.", err)
	}

	if e, r := event.InvolvedObject, response.InvolvedObject; !reflect.DeepEqual(e, r) {
		t.Errorf("%#v != %#v.", e, r)
	}
}

func TestEventList(t *testing.T) {
	ns := api.NamespaceDefault
	objReference := &api.ObjectReference{
		Kind:            "foo",
		Namespace:       ns,
		Name:            "objref1",
		UID:             "uid",
		APIVersion:      "apiv1",
		ResourceVersion: "1",
	}
	timeStamp := util.Now()
	eventList := &api.EventList{
		Items: []api.Event{
			{
				InvolvedObject: *objReference,
				FirstTimestamp: timeStamp,
				LastTimestamp:  timeStamp,
				Count:          1,
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.ResourcePath("events", ns, ""),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: eventList},
	}
	response, err := c.Setup().Events(ns).List(labels.Everything(),
		fields.Everything())

	if err != nil {
		t.Errorf("%#v should be nil.", err)
	}

	if len(response.Items) != 1 {
		t.Errorf("%#v response.Items should have len 1.", response.Items)
	}

	responseEvent := response.Items[0]
	if e, r := eventList.Items[0].InvolvedObject,
		responseEvent.InvolvedObject; !reflect.DeepEqual(e, r) {
		t.Errorf("%#v != %#v.", e, r)
	}
}

func TestEventDelete(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request: testRequest{
			Method: "DELETE",
			Path:   testapi.ResourcePath("events", ns, "foo"),
		},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup().Events(ns).Delete("foo")
	c.Validate(t, nil, err)
}
