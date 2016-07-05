/*
Copyright 2014 The Kubernetes Authors.

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

package validation

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
)

func TestValidateEvent(t *testing.T) {
	table := []struct {
		*api.Event
		valid bool
	}{
		{
			&api.Event{
				ObjectMeta: api.ObjectMeta{
					Name:      "test1",
					Namespace: "foo",
				},
				InvolvedObject: api.ObjectReference{
					Namespace: "bar",
				},
			},
			false,
		}, {
			&api.Event{
				ObjectMeta: api.ObjectMeta{
					Name:      "test1",
					Namespace: "aoeu-_-aoeu",
				},
				InvolvedObject: api.ObjectReference{
					Namespace: "aoeu-_-aoeu",
				},
			},
			false,
		},
	}

	for _, item := range table {
		if e, a := item.valid, len(ValidateEvent(item.Event)) == 0; e != a {
			t.Errorf("%v: expected %v, got %v", item.Event.Name, e, a)
		}
	}
}
