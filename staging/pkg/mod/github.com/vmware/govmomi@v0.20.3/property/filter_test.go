/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package property

import (
	"testing"

	"github.com/vmware/govmomi/vim25/types"
)

func TestMatchProperty(t *testing.T) {
	tests := []struct {
		key  string
		val  types.AnyType
		pass types.AnyType
		fail types.AnyType
	}{
		{"string", "bar", "bar", "foo"},
		{"match", "foo.bar", "foo.*", "foobarbaz"},
		{"moref", types.ManagedObjectReference{Type: "HostSystem", Value: "foo"}, "HostSystem:foo", "bar"}, // implements fmt.Stringer
		{"morefm", types.ManagedObjectReference{Type: "HostSystem", Value: "foo"}, "*foo", "bar"},
		{"morefs", types.ArrayOfManagedObjectReference{ManagedObjectReference: []types.ManagedObjectReference{{Type: "HostSystem", Value: "foo"}}}, "*foo", "bar"},
		{"enum", types.VirtualMachinePowerStatePoweredOn, "poweredOn", "poweredOff"},
		{"int16", int32(16), int32(16), int32(42)},
		{"int32", int32(32), int32(32), int32(42)},
		{"int32s", int32(32), "32", "42"},
		{"int64", int64(64), int64(64), int64(42)},
		{"int64s", int64(64), "64", "42"},
		{"float32", float32(32.32), float32(32.32), float32(42.0)},
		{"float32s", float32(32.32), "32.32", "42.0"},
		{"float64", float64(64.64), float64(64.64), float64(42.0)},
		{"float64s", float64(64.64), "64.64", "42.0"},
	}

	for _, test := range tests {
		p := types.DynamicProperty{Name: test.key, Val: test.val}

		for match, value := range map[bool]types.AnyType{true: test.pass, false: test.fail} {
			result := Filter{test.key: value}.MatchProperty(p)

			if result != match {
				t.Errorf("%s: %t", test.key, result)
			}
		}
	}
}
