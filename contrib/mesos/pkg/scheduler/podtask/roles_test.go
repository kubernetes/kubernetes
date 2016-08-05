/*
Copyright 2015 The Kubernetes Authors.

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

package podtask

import (
	"reflect"
	"testing"
)

func TestFilterRoles(t *testing.T) {
	for i, tt := range []struct {
		roles, want []string
		predicates  []rolePredicate
	}{
		{
			[]string{"role1", "", "role1", "role2", "role3", "role2"},
			[]string{"role1", "role2", "role3"},
			[]rolePredicate{not(emptyRole), not(seenRole())},
		},
		{
			[]string{},
			[]string{},
			[]rolePredicate{not(emptyRole)},
		},
		{
			[]string{""},
			[]string{},
			[]rolePredicate{not(emptyRole)},
		},
		{
			nil,
			[]string{},
			[]rolePredicate{not(emptyRole)},
		},
		{
			[]string{"role1", "role2"},
			[]string{"role1", "role2"},
			nil,
		},
		{
			nil,
			[]string{},
			nil,
		},
	} {
		got := filterRoles(tt.roles, tt.predicates...)

		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("test #%d got %#v want %#v", i, got, tt.want)
		}
	}
}
