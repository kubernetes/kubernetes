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

package registry

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
)

func TestCloudList(t *testing.T) {
	instances := []string{"m1", "m2"}
	fakeCloud := cloudprovider.FakeCloud{
		Machines: instances,
	}
	registry, err := MakeCloudMinionRegistry(&fakeCloud, ".*")
	expectNoError(t, err)

	list, err := registry.List()
	expectNoError(t, err)
	if !reflect.DeepEqual(list, instances) {
		t.Errorf("Unexpected inequality: %#v, %#v", list, instances)
	}
}

func TestCloudContains(t *testing.T) {
	instances := []string{"m1", "m2"}
	fakeCloud := cloudprovider.FakeCloud{
		Machines: instances,
	}
	registry, err := MakeCloudMinionRegistry(&fakeCloud, ".*")
	expectNoError(t, err)

	contains, err := registry.Contains("m1")
	expectNoError(t, err)
	if !contains {
		t.Errorf("Unexpected !contains")
	}

	contains, err = registry.Contains("m100")
	expectNoError(t, err)
	if contains {
		t.Errorf("Unexpected contains")
	}
}

func TestCloudListRegexp(t *testing.T) {
	instances := []string{"m1", "m2", "n1", "n2"}
	fakeCloud := cloudprovider.FakeCloud{
		Machines: instances,
	}
	registry, err := MakeCloudMinionRegistry(&fakeCloud, "m[0-9]+")
	expectNoError(t, err)

	list, err := registry.List()
	expectNoError(t, err)
	expectedList := []string{"m1", "m2"}
	if !reflect.DeepEqual(list, expectedList) {
		t.Errorf("Unexpected inequality: %#v, %#v", list, expectedList)
	}
}
