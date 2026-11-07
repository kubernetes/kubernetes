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

package resourceclaimspec

import (
	"bytes"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/apis/resource"
)

func TestPreserveOpaqueParameters(t *testing.T) {
	for _, tc := range []struct {
		name, oldJSON, newJSON string
		equal                  bool
	}{
		{"identical", `{"a":1}`, `{"a":1}`, true},
		{"whitespace", `{"a": 1}`, `{"a":1}`, true},
		{"object order", `{"z":1,"a":"b"}`, `{"a":"b","z":1}`, true},
		{"nested order", `{"a":[{"z":1,"b":2}]}`, `{ "a": [ {"b":2,"z":1} ] }`, true},
		{"array order", `{"a":[1,2]}`, `{"a":[2,1]}`, false},
		{"value", `{"a":1}`, `{"a":2}`, false},
		{"type", `{"a":"1"}`, `{"a":1}`, false},
		{"missing and null", `{}`, `{"a":null}`, false},
		{"large integer equal", `{"a": 18446744073709551615}`, `{"a":18446744073709551615}`, true},
		{"large integer changed", `{"a":18446744073709551615}`, `{"a":18446744073709551614}`, false},
		{"precise decimal changed", `{"a":0.12345678901234567891}`, `{"a":0.12345678901234567892}`, false},
		{"number spelling", `{"a":1}`, `{"a":1.0}`, false},
		{"exponent spelling", `{"a":1e0}`, `{"a":1}`, false},
		{"negative zero", `{"a":-0}`, `{"a":0}`, false},
		{"invalid old", `{"a":1`, `{"a":1}`, false},
		{"invalid new", `{"a":1}`, `{"a":1`, false},
		{"trailing value", `{"a":1}`, `{"a":1} {}`, false},
		{"duplicate whitespace", `{"a": 1, "a": 2}`, `{"a":1,"a":2}`, true},
		{"duplicate object order", `{"b":3,"a":1,"a":2}`, `{"a":1,"a":2,"b":3}`, true},
		{"duplicate value order", `{"a":1,"a":2}`, `{"a":2,"a":1}`, true},
		{"duplicate old", `{"a":1,"a":2}`, `{"a":2}`, false},
		{"duplicate new", `{"a":2}`, `{"a":1,"a":2}`, false},
		{"duplicate nested", `{"a":{"b":1,"b":2}}`, `{"a":{"b":2}}`, false},
		{"escaped duplicate", `{"a":1,"\u0061":2}`, `{"a":2}`, false},
		{"invalid utf8", "{\"a\":\"\xff\"}", `{"a":"�"}`, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			old := opaqueSpec(tc.oldJSON)
			updated := opaqueSpec(tc.newJSON)
			PreserveOpaqueParameters(updated, old)
			want := tc.newJSON
			if tc.equal {
				want = tc.oldJSON
			}
			if got := string(updated.Devices.Config[0].Opaque.Parameters.Raw); got != want {
				t.Fatalf("parameters = %q, want %q", got, want)
			}
			if got := string(old.Devices.Config[0].Opaque.Parameters.Raw); got != tc.oldJSON {
				t.Fatalf("old parameters mutated: %q", got)
			}
		})
	}
}

func TestPreserveOpaqueParametersConfigurationIdentity(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*resource.ResourceClaimSpec)
	}{
		{"driver", func(s *resource.ResourceClaimSpec) { s.Devices.Config[0].Opaque.Driver = "other.example.com" }},
		{"requests", func(s *resource.ResourceClaimSpec) { s.Devices.Config[0].Requests = []string{"other"} }},
		{"decoded object", func(s *resource.ResourceClaimSpec) {
			s.Devices.Config[0].Opaque.Parameters.Object = &resource.ResourceClaim{}
		}},
		{"missing opaque", func(s *resource.ResourceClaimSpec) { s.Devices.Config[0].Opaque = nil }},
		{"removed config", func(s *resource.ResourceClaimSpec) { s.Devices.Config = nil }},
		{"added config", func(s *resource.ResourceClaimSpec) { s.Devices.Config = append(s.Devices.Config, s.Devices.Config[0]) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			old := opaqueSpec(`{"a": 1}`)
			updated := opaqueSpec(`{"a":1}`)
			tc.mutate(updated)
			before := updated.DeepCopy()
			PreserveOpaqueParameters(updated, old)
			if !reflect.DeepEqual(updated, before) {
				t.Fatal("changed configuration was modified")
			}
		})
	}
	t.Run("multiple configurations", func(t *testing.T) {
		old := opaqueSpec(`{"a": 1}`)
		second := opaqueSpec(`{"b": 2}`).Devices.Config[0]
		second.Opaque.Driver = "second.example.com"
		old.Devices.Config = append(old.Devices.Config, second)
		updated := old.DeepCopy()
		for i, raw := range []string{`{"a":1}`, `{"b":2}`} {
			updated.Devices.Config[i].Opaque.Parameters.Raw = []byte(raw)
		}
		PreserveOpaqueParameters(updated, old)
		for i := range old.Devices.Config {
			if !bytes.Equal(updated.Devices.Config[i].Opaque.Parameters.Raw, old.Devices.Config[i].Opaque.Parameters.Raw) {
				t.Fatalf("configuration %d did not preserve original bytes", i)
			}
		}
		updated = old.DeepCopy()
		updated.Devices.Config[0], updated.Devices.Config[1] = updated.Devices.Config[1], updated.Devices.Config[0]
		before := updated.DeepCopy()
		PreserveOpaqueParameters(updated, old)
		if !reflect.DeepEqual(updated, before) {
			t.Fatal("reordered configurations were modified")
		}
	})
}

func opaqueSpec(raw string) *resource.ResourceClaimSpec {
	return &resource.ResourceClaimSpec{Devices: resource.DeviceClaim{Config: []resource.DeviceClaimConfiguration{{
		Requests: []string{"req-0"},
		DeviceConfiguration: resource.DeviceConfiguration{Opaque: &resource.OpaqueDeviceConfiguration{
			Driver: "driver.example.com", Parameters: runtime.RawExtension{Raw: []byte(raw)},
		}},
	}}}}
}
