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

package wholepkg

import (
	"reflect"
	"testing"

	fuzz "github.com/google/gofuzz"
)

func TestDeepCopy(t *testing.T) {
	x := Struct_Primitives{}
	y := Struct_Primitives{}

	if !reflect.DeepEqual(&x, &y) {
		t.Errorf("objects should be equal to start, but are not")
	}

	fuzzer := fuzz.New()
	fuzzer.Fuzz(&x)
	fuzzer.Fuzz(&y)

	if reflect.DeepEqual(&x, &y) {
		t.Errorf("objects should not be equal, but are")
	}

	if err := DeepCopy_wholepkg_Struct_Primitives(&x, &y, nil); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(&x, &y) {
		t.Errorf("objects should be equal, but are not")
	}
}
