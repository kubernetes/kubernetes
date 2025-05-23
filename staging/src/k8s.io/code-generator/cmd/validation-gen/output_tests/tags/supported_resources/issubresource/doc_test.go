/*
Copyright 2024 The Kubernetes Authors.

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

package issubresource

import (
	"fmt"
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestRegisterValidations(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	t1 := &T1{}
	st.Value(t1).ExpectInvalid(
		field.InternalError(nil, fmt.Errorf("no validation found for %T, subresource: %v", t1, "/")),
	)

	st.Value(t1).Subresources([]string{"scale"}).ExpectValid()

	st.Value(t1).Subresources([]string{"unknown"}).ExpectInvalid(
		field.InternalError(nil, fmt.Errorf("no validation found for %T, subresource: %v", t1, "/unknown")),
	)
}
