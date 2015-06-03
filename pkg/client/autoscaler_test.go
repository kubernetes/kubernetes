/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

func TestAutoScalerList(t *testing.T) {
	ns := api.NamespaceDefault
	names := []string{"foo", "bar", "baz"}

	namesMap := make(map[string]int, len(names))
	asList := &api.AutoScalerList{Items: make([]api.AutoScaler, len(names))}

	for k, v := range names {
		namesMap[v] = 1
		asList.Items[k] = api.AutoScaler{ObjectMeta: api.ObjectMeta{Name: v}}
	}

	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.ResourcePath("autoscalers", ns, ""),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: asList},
	}

	response, err := c.Setup().AutoScalers(ns).List(labels.Everything(), fields.Everything())
	c.Validate(t, response, err)

	if err != nil {
		t.Errorf("list error %#v where none was expected.", err)
	}

	if len(response.Items) != len(asList.Items) {
		t.Errorf("got a list of %v items, expected a list of %v items.",
			len(response.Items), len(asList.Items))
	}

	for _, as := range response.Items {
		if _, ok := namesMap[as.Name]; !ok {
			t.Errorf("unexpected autoscaler %#v", as.Name)
		}
	}
}

func TestAutoscalerCreate(t *testing.T) {
	ns := "autoscaler-create-test"
	as1 := &api.AutoScaler{
		ObjectMeta: api.ObjectMeta{Name: "crAter"},

		Spec: api.AutoScalerSpec{
			MinAutoScaleCount: 1,
			MaxAutoScaleCount: 10,
			TargetSelector:    map[string]string{"foo": "bar"},
			MonitoringSources: []string{
				"cadvisor",
				"influxdb",
				"statsd",
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "POST",
			Path:   testapi.ResourcePath("autoscalers", ns, ""),
			Body:   as1,
		},
		Response: Response{StatusCode: 200, Body: as1},
	}

	response, err := c.Setup().AutoScalers(ns).Create(as1)
	c.Validate(t, response, err)

	if err != nil {
		t.Errorf("create error %#v where none was expected.", err)
	}

	if response.Name != as1.Name {
		t.Errorf("create returned autoscaler %#v, expected %#v.",
			response.Name, as1.Name)
	}

	if response.Spec.MinAutoScaleCount != as1.Spec.MinAutoScaleCount {
		t.Errorf("create returned MinAutoScaleCount %#v, expected %#v.",
			response.Spec.MinAutoScaleCount,
			as1.Spec.MinAutoScaleCount)
	}

	if response.Spec.MaxAutoScaleCount != as1.Spec.MaxAutoScaleCount {
		t.Errorf("create returned MaxAutoScaleCount %#v, expected %#v.",
			response.Spec.MaxAutoScaleCount,
			as1.Spec.MaxAutoScaleCount)
	}

	if len(response.Spec.MonitoringSources) != len(as1.Spec.MonitoringSources) {
		t.Errorf("create returned MonitoringSources %#v, expected %#v.",
			response.Spec.MonitoringSources,
			as1.Spec.MonitoringSources)
	}
}

func TestAutoScalerGet(t *testing.T) {
	ns := api.NamespaceDefault
	name := "auto-scaler-getty"
	as1 := &api.AutoScaler{
		ObjectMeta: api.ObjectMeta{Name: name},

		Spec: api.AutoScalerSpec{
			MaxAutoScaleCount: 42,
			TargetSelector:    map[string]string{"as": "one"},
			MonitoringSources: []string{"graphite"},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.ResourcePath("autoscalers", ns, name),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: as1},
	}

	response, err := c.Setup().AutoScalers(ns).Get(name)
	c.Validate(t, response, err)

	if err != nil {
		t.Errorf("get error %#v where none was expected.", err)
	}

	if response.Name != as1.Name {
		t.Errorf("got autoscaler %#v, expected %#v.",
			response.Name, as1.Name)
	}

	if response.Spec.MaxAutoScaleCount != as1.Spec.MaxAutoScaleCount {
		t.Errorf("create returned MaxAutoScaleCount %#v, expected %#v.",
			response.Spec.MaxAutoScaleCount,
			as1.Spec.MaxAutoScaleCount)
	}

	if len(response.Spec.MonitoringSources) != len(as1.Spec.MonitoringSources) {
		t.Errorf("create returned MonitoringSources %#v, expected %#v.",
			response.Spec.MonitoringSources,
			as1.Spec.MonitoringSources)
	}
}

func TestAutoScalerUpdate(t *testing.T) {
	ns := "upd8"
	name := "autoscaler-update-test"
	as1 := &api.AutoScaler{
		ObjectMeta: api.ObjectMeta{Name: name},

		Spec: api.AutoScalerSpec{
			MaxAutoScaleCount: 123,
			TargetSelector:    map[string]string{"as": "one"},
			MonitoringSources: []string{"cadvisor"},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "PUT",
			Path:   testapi.ResourcePath("autoscalers", ns, name),
		},
		Response: Response{StatusCode: 200, Body: as1},
	}

	response, err := c.Setup().AutoScalers(ns).Update(as1)
	c.Validate(t, response, err)

	if response.Spec.MaxAutoScaleCount != as1.Spec.MaxAutoScaleCount {
		t.Errorf("create returned MaxAutoScaleCount %#v, expected %#v.",
			response.Spec.MaxAutoScaleCount,
			as1.Spec.MaxAutoScaleCount)
	}
}

func TestAutoScalerDelete(t *testing.T) {
	ns := api.NamespaceDefault
	name := "deli"
	c := &testClient{
		Request: testRequest{
			Method: "DELETE",
			Path:   testapi.ResourcePath("autoscalers", ns, name),
		},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup().AutoScalers(ns).Delete(name)
	c.Validate(t, nil, err)
}

func TestAutoScalerWatch(t *testing.T) {
	ns := api.NamespaceAll
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   "/api/" + testapi.Version() + "/watch/autoscalers",
			Query:  url.Values{"resourceVersion": []string{}},
		},
		Response: Response{StatusCode: 200},
	}
	_, err := c.Setup().AutoScalers(ns).Watch(labels.Everything(), fields.Everything(), "")
	c.Validate(t, nil, err)
}
