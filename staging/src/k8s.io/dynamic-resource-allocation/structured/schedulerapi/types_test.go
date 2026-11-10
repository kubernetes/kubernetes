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

package schedulerapi

import (
	"testing"

	"github.com/stretchr/testify/require"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	draapi "k8s.io/dynamic-resource-allocation/api"
)

func TestNewDeviceConsumedCapacity(t *testing.T) {
	deviceID := DeviceID{Driver: draapi.MakeUniqueString("driver-a"), Pool: draapi.MakeUniqueString("pool-1"), Device: draapi.MakeUniqueString("device-1")}
	one := resource.MustParse("1")
	two := resource.MustParse("2")
	four := resource.MustParse("4")
	u := draapi.MakeUniqueString

	tests := map[string]struct {
		consumedCapacity map[resourceapi.QualifiedName]resource.Quantity
		want             DeviceConsumedCapacity
	}{
		"empty": {
			consumedCapacity: map[resourceapi.QualifiedName]resource.Quantity{},
			want:             DeviceConsumedCapacity{DeviceID: deviceID, ConsumedCapacity: ConsumedCapacity{}},
		},
		"unqualified-only": {
			consumedCapacity: map[resourceapi.QualifiedName]resource.Quantity{"cap": one},
			want:             DeviceConsumedCapacity{DeviceID: deviceID, ConsumedCapacity: ConsumedCapacity{{Domain: deviceID.Driver, Identifier: u("cap")}: &one}},
		},
		"driver-qualified-only": {
			consumedCapacity: map[resourceapi.QualifiedName]resource.Quantity{"driver-a/cap": one},
			want:             DeviceConsumedCapacity{DeviceID: deviceID, ConsumedCapacity: ConsumedCapacity{{Domain: deviceID.Driver, Identifier: u("cap")}: &one}},
		},
		"foreign-domain-qualified": {
			consumedCapacity: map[resourceapi.QualifiedName]resource.Quantity{"example.com/cap": one},
			want:             DeviceConsumedCapacity{DeviceID: deviceID, ConsumedCapacity: ConsumedCapacity{{Domain: u("example.com"), Identifier: u("cap")}: &one}},
		},
		"unqualified-and-driver-qualified-same-identifier-deterministic": {
			// A 1.37 allocator could have persisted both spellings for the same
			// device and capacity. The driver-qualified entry must win
			// regardless of map iteration order, matching buildCapacity in
			// k8s.io/dynamic-resource-allocation/cel.
			consumedCapacity: map[resourceapi.QualifiedName]resource.Quantity{"cap": one, "driver-a/cap": two},
			want:             DeviceConsumedCapacity{DeviceID: deviceID, ConsumedCapacity: ConsumedCapacity{{Domain: deviceID.Driver, Identifier: u("cap")}: &two}},
		},
		"unrelated-unqualified-and-qualified-identifiers-do-not-interfere": {
			consumedCapacity: map[resourceapi.QualifiedName]resource.Quantity{"cap": one, "driver-a/other-cap": four},
			want: DeviceConsumedCapacity{DeviceID: deviceID, ConsumedCapacity: ConsumedCapacity{
				{Domain: deviceID.Driver, Identifier: u("cap")}:       &one,
				{Domain: deviceID.Driver, Identifier: u("other-cap")}: &four,
			}},
		},
		"unqualified-and-foreign-domain-same-identifier-do-not-collide": {
			consumedCapacity: map[resourceapi.QualifiedName]resource.Quantity{"cap": one, "example.com/cap": two},
			want: DeviceConsumedCapacity{DeviceID: deviceID, ConsumedCapacity: ConsumedCapacity{
				{Domain: deviceID.Driver, Identifier: u("cap")}:  &one,
				{Domain: u("example.com"), Identifier: u("cap")}: &two,
			}},
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			got := NewDeviceConsumedCapacity(deviceID, tt.consumedCapacity)
			require.Equal(t, tt.want, got)
		})
	}
}
