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

package framework

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
)

func TestRCNumber(t *testing.T) {
	pod := func(name string) *api.Pod {
		return &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: name,
			},
		}
	}

	source := NewFakeControllerSource()
	source.Add(pod("foo"))
	source.Modify(pod("foo"))
	source.Modify(pod("foo"))

	w, err := source.Watch("1")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer w.Stop()
	got := <-w.ResultChan()
	if e, a := "2", got.Object.(*api.Pod).ObjectMeta.ResourceVersion; e != a {
		t.Errorf("wanted %v, got %v", e, a)
	}
}
