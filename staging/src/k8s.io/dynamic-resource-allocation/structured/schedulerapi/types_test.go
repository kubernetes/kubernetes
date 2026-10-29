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

func TestDeviceAffinityStateCollectionZeroValueIsClean(t *testing.T) {
	if AffinityStatusClean != 0 {
		t.Errorf("AffinityStatusClean = %d, want 0", AffinityStatusClean)
	}
	for name, state := range map[string]AllocatedState{
		"zero state": {},
		"empty affinity map": {
			DeviceSharingAffinity: DeviceAffinityStateCollection{},
		},
	} {
		t.Run(name, func(t *testing.T) {
			got := state.DeviceSharingAffinity[MakeDeviceID("driver", "pool", "device")]
			if got.Status != AffinityStatusClean || got.LockedAffinity != nil {
				t.Errorf("absent device state = %#v, want Clean with nil lock", got)
			}
		})
	}
}

// TestDeviceAffinityStateCollectionCloneIsolation checks that DeviceAffinityState
// and DeviceAffinityStateCollection clones preserve values and nil/empty distinctions.
// Mutations must not leak between source and clone or between different devices' lock maps.
func TestDeviceAffinityStateCollectionCloneIsolation(t *testing.T) {
	t.Run("state", func(t *testing.T) {
		for _, status := range []AffinityStatus{AffinityStatusClean, AffinityStatusUnreconstructable} {
			source := DeviceAffinityState{Status: status}
			got := source.Clone()
			require.Equal(t, source, got)
		}
		source := DeviceAffinityState{Status: AffinityStatusLocked, LockedAffinity: map[string]string{"subnet": "x", "tenant": "a"}}
		got := source.Clone()
		require.Equal(t, source, got)

		got.LockedAffinity["subnet"] = "y"
		delete(got.LockedAffinity, "tenant")
		require.Equal(t, DeviceAffinityState{
			Status:         AffinityStatusLocked,
			LockedAffinity: map[string]string{"subnet": "x", "tenant": "a"},
		}, source, "mutating clone changed source")

		source.LockedAffinity["new"] = "source-only"
		require.NotContains(t, got.LockedAffinity, "new", "mutating source changed clone")
	})
	t.Run("device states", func(t *testing.T) {
		var nilStates DeviceAffinityStateCollection
		require.Nil(t, nilStates.Clone())

		emptySource := DeviceAffinityStateCollection{}
		empty := emptySource.Clone()
		require.Equal(t, DeviceAffinityStateCollection{}, empty)
		device := MakeDeviceID("driver", "pool", "device")
		empty[device] = DeviceAffinityState{Status: AffinityStatusUnreconstructable}
		require.Empty(t, emptySource, "empty device states clone aliases source")

		otherDevice := MakeDeviceID("driver", "pool", "other-device")
		unknown := MakeDeviceID("driver", "pool", "unknown")
		source := DeviceAffinityStateCollection{
			device:      {Status: AffinityStatusLocked, LockedAffinity: map[string]string{"subnet": "x"}},
			otherDevice: {Status: AffinityStatusLocked, LockedAffinity: map[string]string{"subnet": "z"}},
			unknown:     {Status: AffinityStatusUnreconstructable},
		}
		got := source.Clone()
		require.Equal(t, source, got)

		got[device].LockedAffinity["subnet"] = "y"
		require.Equal(t, map[string]string{"subnet": "z"}, got[otherDevice].LockedAffinity, "cloned devices share a lock map")
		got[otherDevice].LockedAffinity["subnet"] = "w"
		require.Equal(t, map[string]string{"subnet": "y"}, got[device].LockedAffinity, "cloned devices share a lock map")
		delete(got, unknown)
		require.Equal(t, DeviceAffinityStateCollection{
			device:      {Status: AffinityStatusLocked, LockedAffinity: map[string]string{"subnet": "x"}},
			otherDevice: {Status: AffinityStatusLocked, LockedAffinity: map[string]string{"subnet": "z"}},
			unknown:     {Status: AffinityStatusUnreconstructable},
		}, source, "device states clone aliases source")

		delete(source, device)
		require.Contains(t, got, device, "source deletion affected device states clone")
	})
}

func TestSharingAffinityStatusString(t *testing.T) {
	for _, tc := range []struct {
		status AffinityStatus
		want   string
	}{
		{AffinityStatusClean, "Clean"},
		{AffinityStatusLocked, "Locked"},
		{AffinityStatusUnreconstructable, "Unreconstructable"},
		{AffinityStatus(-1), "AffinityStatus(-1)"},
		{AffinityStatus(42), "AffinityStatus(42)"},
	} {
		t.Run(tc.want, func(t *testing.T) {
			if got := tc.status.String(); got != tc.want {
				t.Errorf("status %d String() = %q, want %q", tc.status, got, tc.want)
			}
		})
	}
}

func TestPoolIDString(t *testing.T) {
	for name, tc := range map[string]struct {
		pool PoolID
		want string
	}{
		"populated": {
			pool: PoolID{Driver: "driver", Pool: "pool"},
			want: "driver/pool",
		},
		"zero value": {
			want: "/",
		},
	} {
		t.Run(name, func(t *testing.T) {
			require.Equal(t, tc.want, tc.pool.String())
		})
	}
}

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
