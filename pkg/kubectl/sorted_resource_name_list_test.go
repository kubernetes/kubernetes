/*
Copyright 2016 The Kubernetes Authors.

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

package kubectl

import (
	"reflect"
	"sort"
	"testing"

	"k8s.io/kubernetes/pkg/api"
)

func TestSortableResourceNamesSorting(t *testing.T) {
	want := SortableResourceNames{
		api.ResourceName(""),
		api.ResourceName("42"),
		api.ResourceName("bar"),
		api.ResourceName("foo"),
		api.ResourceName("foo"),
		api.ResourceName("foobar"),
	}

	in := SortableResourceNames{
		api.ResourceName("foo"),
		api.ResourceName("42"),
		api.ResourceName("foobar"),
		api.ResourceName("foo"),
		api.ResourceName("bar"),
		api.ResourceName(""),
	}

	sort.Sort(in)
	if !reflect.DeepEqual(in, want) {
		t.Errorf("got %v, want %v", in, want)
	}
}
