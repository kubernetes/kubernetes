/*
Copyright 2021 The Kubernetes Authors.

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

package state

import (
	"reflect"
	"testing"

	"k8s.io/utils/cpuset"
)

func TestClone(t *testing.T) {
	expect := ContainerCPUAssignments{
		"pod": map[string]cpuset.CPUSet{
			"container1": cpuset.New(4, 5, 6),
			"container2": cpuset.New(1, 2, 3),
		},
	}
	actual := expect.Clone()
	if &expect == &actual || !reflect.DeepEqual(expect, actual) {
		t.Fail()
	}
}
