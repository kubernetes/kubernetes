/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/api/resource"
)

func TestSortableResourceNamesSorting(t *testing.T) {
	want := SortableResourceNames{
		resource.Name(""),
		resource.Name("42"),
		resource.Name("bar"),
		resource.Name("foo"),
		resource.Name("foo"),
		resource.Name("foobar"),
	}

	in := SortableResourceNames{
		resource.Name("foo"),
		resource.Name("42"),
		resource.Name("foobar"),
		resource.Name("foo"),
		resource.Name("bar"),
		resource.Name(""),
	}

	sort.Sort(in)
	if !reflect.DeepEqual(in, want) {
		t.Errorf("got %v, want %v", in, want)
	}
}
