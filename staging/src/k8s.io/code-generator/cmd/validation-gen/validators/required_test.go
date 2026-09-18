/*
Copyright The Kubernetes Authors.

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

package validators

import (
	"strings"
	"testing"

	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

// A field context is documented to carry the struct member it describes.
// A caller which omits it must get an error, not a nil dereference.
func TestRequirednessWithoutMember(t *testing.T) {
	fieldType := &types.Type{Name: types.Name{Name: "int"}, Kind: types.Builtin}
	tv := requirednessTagValidator{mode: requirednessOptional}
	context := Context{Scope: ScopeField, Type: fieldType}

	_, err := tv.GetValidations(context, codetags.Tag{Name: string(requirednessOptional)})
	if err == nil {
		t.Fatal("GetValidations() succeeded, want an error")
	}
	if !strings.Contains(err.Error(), "no member") {
		t.Errorf("GetValidations() error = %v, want it to mention a missing member", err)
	}
}
