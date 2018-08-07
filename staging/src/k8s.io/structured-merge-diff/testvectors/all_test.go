/*
Copyright 2018 The Kubernetes Authors.

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

package testvectors

import (
	"testing"
)

func TestAll(t *testing.T) {
	RunAllVectors(t, implementation{})
}

type implementation struct{}

func (implementation) Test(t *testing.T, v *Vector, s SchemaDefinition) {
	t.Parallel()
	t.Skip("not implemented yet")

	var implementationUnderTest inTreeImplementation
	tmp := implementationUnderTest.diff(s, v.NewObject, v.LastObject)
	got := implementationUnderTest.merge(s, v.LiveObject, tmp)

	if got != v.ExpectedObject {
		t.Errorf("%v: did not get expected object", v.Name)
	}
}

// TODO: delete this. It's here as an illustration of the approach and isn't
// necessary after we publish the necessary functions.
type inTreeImplementation interface {
	diff(s SchemaDefinition, a, b YAMLObject) YAMLObject
	merge(s SchemaDefinition, a, b YAMLObject) YAMLObject
}
