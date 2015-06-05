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

package apiserver

import (
	stderrs "errors"
	"net/http"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
)

func TestErrorsToAPIStatus(t *testing.T) {
	cases := map[error]api.Status{
		errors.NewNotFound("foo", "bar"): {
			Status:  api.StatusFailure,
			Code:    http.StatusNotFound,
			Reason:  api.StatusReasonNotFound,
			Message: "foo \"bar\" not found",
			Details: &api.StatusDetails{
				Kind: "foo",
				Name: "bar",
			},
		},
		errors.NewAlreadyExists("foo", "bar"): {
			Status:  api.StatusFailure,
			Code:    http.StatusConflict,
			Reason:  "AlreadyExists",
			Message: "foo \"bar\" already exists",
			Details: &api.StatusDetails{
				Kind: "foo",
				Name: "bar",
			},
		},
		errors.NewConflict("foo", "bar", stderrs.New("failure")): {
			Status:  api.StatusFailure,
			Code:    http.StatusConflict,
			Reason:  "Conflict",
			Message: "foo \"bar\" cannot be updated: failure",
			Details: &api.StatusDetails{
				Kind: "foo",
				Name: "bar",
			},
		},
	}
	for k, v := range cases {
		actual := errToAPIStatus(k)
		if !reflect.DeepEqual(actual, &v) {
			t.Errorf("%s: Expected %#v, Got %#v", k, v, actual)
		}
	}
}
