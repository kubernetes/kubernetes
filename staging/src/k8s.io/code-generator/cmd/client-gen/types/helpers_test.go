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

package types

import (
	"reflect"
	"sort"
	"testing"
)

func TestVersionSort(t *testing.T) {
	unsortedVersions := []string{"v4beta1", "v2beta1", "v2alpha1", "v3", "v1"}
	expected := []string{"v2alpha1", "v2beta1", "v4beta1", "v1", "v3"}
	sort.Sort(sortableSliceOfVersions(unsortedVersions))
	if !reflect.DeepEqual(unsortedVersions, expected) {
		t.Errorf("expected %#v\ngot %#v", expected, unsortedVersions)
	}
}
