/*
Copyright 2014 Google Inc. All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func TestEventSearch(t *testing.T) {
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   "/events",
			Query: url.Values{
				"fields": []string{"involvedObject.kind=Pod,involvedObject.name=foo,involvedObject.namespace=baz"},
				"labels": []string{},
			},
		},
		Response: Response{StatusCode: 200, Body: &api.EventList{}},
	}
	eventList, err := c.Setup().Events("").Search(
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
		//namespace: namespace{"default"},
		Status:         "running",
		InvolvedObject: *objReference,
		Timestamp:      timeStamp,
	}
	c := &testClient{
		Request: testRequest{
			Method: "POST",
			Path:   "/events",
			Body:   event,
		},
		Response: Response{StatusCode: 200, Body: event},
	}

	response, err := c.Setup().Events("").Create(event)

	if err != nil {
		t.Errorf("%#v should be nil.", err)
	}

	if e, a := *objReference, response.InvolvedObject; !reflect.DeepEqual(e, a) {
		t.Errorf("%#v != %#v.", e, a)
	}
}
