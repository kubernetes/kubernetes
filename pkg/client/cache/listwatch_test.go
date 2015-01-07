/*
Copyright 2015 Google Inc. All rights reserved.

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

package cache

import (
	"net/http/httptest"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func parseSelectorOrDie(s string) labels.Selector {
	selector, err := labels.ParseSelector(s)
	if err != nil {
		panic(err)
	}
	return selector
}

func TestListWatchesCanList(t *testing.T) {
	table := []struct {
		location string
		lw       ListWatch
	}{
		// Minion
		{
			location: "/api/" + testapi.Version() + "/minions",
			lw: ListWatch{
				FieldSelector: parseSelectorOrDie(""),
				Resource:      "minions",
			},
		},
		// pod with "assigned" field selector.
		{
			location: "/api/" + testapi.Version() + "/pods?fields=DesiredState.Host%3D",
			lw: ListWatch{
				FieldSelector: labels.Set{"DesiredState.Host": ""}.AsSelector(),
				Resource:      "pods",
			},
		},
	}

	for _, item := range table {
		handler := util.FakeHandler{
			StatusCode:   500,
			ResponseBody: "",
			T:            t,
		}
		server := httptest.NewServer(&handler)
		defer server.Close()
		item.lw.Client = client.NewOrDie(&client.Config{Host: server.URL, Version: testapi.Version()})
		// This test merely tests that the correct request is made.
		item.lw.List()
		handler.ValidateRequest(t, item.location, "GET", nil)
	}
}

func TestListWatchesCanWatch(t *testing.T) {
	table := []struct {
		rv       string
		location string
		lw       ListWatch
	}{
		// Minion
		{
			location: "/api/" + testapi.Version() + "/watch/minions?resourceVersion=",
			rv:       "",
			lw: ListWatch{
				FieldSelector: parseSelectorOrDie(""),
				Resource:      "minions",
			},
		},
		{
			location: "/api/" + testapi.Version() + "/watch/minions?resourceVersion=42",
			rv:       "42",
			lw: ListWatch{
				FieldSelector: parseSelectorOrDie(""),
				Resource:      "minions",
			},
		},
		// pod with "assigned" field selector.
		{
			location: "/api/" + testapi.Version() + "/watch/pods?fields=DesiredState.Host%3D&resourceVersion=0",
			rv:       "0",
			lw: ListWatch{
				FieldSelector: labels.Set{"DesiredState.Host": ""}.AsSelector(),
				Resource:      "pods",
			},
		},
	}

	for _, item := range table {
		handler := util.FakeHandler{
			StatusCode:   500,
			ResponseBody: "",
			T:            t,
		}
		server := httptest.NewServer(&handler)
		defer server.Close()
		item.lw.Client = client.NewOrDie(&client.Config{Host: server.URL, Version: testapi.Version()})

		// This test merely tests that the correct request is made.
		item.lw.Watch(item.rv)
		handler.ValidateRequest(t, item.location, "GET", nil)
	}
}
