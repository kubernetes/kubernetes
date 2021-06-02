/*
Copyright 2017 The Kubernetes Authors.

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

package slice

import (
	"reflect"
	"testing"
)

func TestSortInts64(t *testing.T) {
	src := []int64{10, 1, 2, 3, 4, 5, 6}
	expected := []int64{1, 2, 3, 4, 5, 6, 10}
	SortInts64(src)
	if !reflect.DeepEqual(src, expected) {
		t.Errorf("func Ints64 didnt sort correctly, %v !- %v", src, expected)
	}
}
