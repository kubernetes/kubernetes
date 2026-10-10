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

package state

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
	"k8s.io/utils/ptr"
)

// Pods written to a V2 checkpoint are read back unchanged, in UID order.
func TestCheckpointV2_RoundTrip(t *testing.T) {
	tests := []struct {
		name string
		list *v1.PodList
		want []v1.Pod
	}{
		{name: "nil list"},
		{name: "no pods", list: &v1.PodList{}},
		{
			name: "pod with every kind of container, pod-level resources and volumes",
			list: &v1.PodList{Items: []v1.Pod{newFullPod("uid-1")}},
			want: []v1.Pod{newFullPod("uid-1")},
		},
		{
			name: "pods come back sorted by UID",
			list: &v1.PodList{Items: []v1.Pod{newFullPod("uid-3"), newFullPod("uid-1"), newFullPod("uid-2")}},
			want: []v1.Pod{newFullPod("uid-1"), newFullPod("uid-2"), newFullPod("uid-3")},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			given := tt.list.DeepCopy()

			cp, err := NewCheckpointV2(tt.list)
			require.NoError(t, err)
			assert.Equal(t, checkpointVersionV2, cp.Version)
			requireNoDiff(t, given, tt.list, "change to the list passed to NewCheckpointV2")

			// Go through the file store, so that the JSON encoding and the checksum are covered too.
			manager, err := checkpointmanager.NewCheckpointManager(t.TempDir())
			require.NoError(t, err)
			require.NoError(t, manager.CreateCheckpoint("checkpoint", cp))
			restored := &Checkpoint{}
			require.NoError(t, manager.GetCheckpoint("checkpoint", restored))

			got, migrated, err := restored.GetPodList()
			require.NoError(t, err)
			assert.False(t, migrated, "a V2 checkpoint must not be treated as V1")
			requireNoDiff(t, tt.want, got.Items, "restored pods")
		})
	}
}

// Identical pods must encode to identical bytes whatever order they are given in, because callers
// compare checksums to skip rewriting an unchanged checkpoint.
func TestCheckpointV2_DeterministicEncoding(t *testing.T) {
	// Each order lists, by UID, the sequence in which the pods are passed in.
	orders := []string{"abc", "acb", "bac", "bca", "cab", "cba"}

	encode := func(t *testing.T, order string) *Checkpoint {
		list := &v1.PodList{}
		for _, uid := range order {
			list.Items = append(list.Items, newFullPod(types.UID(string(uid))))
		}
		cp, err := NewCheckpointV2(list)
		require.NoError(t, err)
		return cp
	}

	want := encode(t, "abc")
	for _, order := range orders {
		t.Run(order, func(t *testing.T) {
			// Map iteration order is randomized on every iteration, so an encoding that depends on it
			// would only differ some of the time.
			for range 10 {
				got := encode(t, order)
				require.Equal(t, want.Data, got.Data)
				require.Equal(t, want.Checksum, got.Checksum)
			}
		})
	}
}

// After a rollback, a kubelet that predates V2 reads the legacy entries and ignores everything else,
// so they must describe the pods the way it expects.
func TestCheckpointV2_LegacyEntries(t *testing.T) {
	tests := []struct {
		name string
		pods []v1.Pod
		want PodResourceInfoMap
	}{
		{name: "no pods"},
		{
			name: "resources of every kind of container, keyed by container name",
			pods: []v1.Pod{podWithSpec("uid-1", v1.PodSpec{
				InitContainers:      []v1.Container{container("init", "100m", "64Mi")},
				Containers:          []v1.Container{container("app", "250m", "256Mi")},
				EphemeralContainers: []v1.EphemeralContainer{ephemeralContainer("debug", "10m", "8Mi")},
			})},
			want: PodResourceInfoMap{"uid-1": {ContainerResources: map[string]v1.ResourceRequirements{
				"init":  requirements("100m", "64Mi"),
				"app":   requirements("250m", "256Mi"),
				"debug": requirements("10m", "8Mi"),
			}}},
		},
		{
			name: "pod-level resources",
			pods: []v1.Pod{podWithSpec("uid-1", v1.PodSpec{Resources: ptr.To(requirements("500m", "512Mi"))})},
			want: PodResourceInfoMap{"uid-1": {PodLevelResources: ptr.To(requirements("500m", "512Mi"))}},
		},
		{
			name: "emptyDir size limits, keyed by volume name",
			pods: []v1.Pod{podWithSpec("uid-1", v1.PodSpec{Volumes: []v1.Volume{
				emptyDirVolume("mem-a", v1.StorageMediumMemory, quantity("64Mi")),
				emptyDirVolume("mem-b", v1.StorageMediumMemory, quantity("128Mi")),
			}})},
			want: PodResourceInfoMap{"uid-1": {EmptyDirVolumeLimits: map[string]*resource.Quantity{
				"mem-a": quantity("64Mi"),
				"mem-b": quantity("128Mi"),
			}}},
		},
		{
			// The old kubelet dereferences the limit it finds for a volume, so recording none would
			// make it panic.
			name: "emptyDir without a size limit is left out",
			pods: []v1.Pod{podWithSpec("uid-1", v1.PodSpec{Volumes: []v1.Volume{
				emptyDirVolume("mem", v1.StorageMediumMemory, nil),
			}})},
			want: PodResourceInfoMap{"uid-1": {}},
		},
		{
			name: "volumes other than emptyDir are left out",
			pods: []v1.Pod{podWithSpec("uid-1", v1.PodSpec{Volumes: []v1.Volume{{
				Name:         "config",
				VolumeSource: v1.VolumeSource{ConfigMap: &v1.ConfigMapVolumeSource{}},
			}}})},
			want: PodResourceInfoMap{"uid-1": {}},
		},
		{
			name: "pods are keyed by UID",
			pods: []v1.Pod{
				podWithSpec("uid-1", v1.PodSpec{Containers: []v1.Container{container("app", "100m", "64Mi")}}),
				podWithSpec("uid-2", v1.PodSpec{Containers: []v1.Container{container("app", "200m", "128Mi")}}),
			},
			want: PodResourceInfoMap{
				"uid-1": {ContainerResources: map[string]v1.ResourceRequirements{"app": requirements("100m", "64Mi")}},
				"uid-2": {ContainerResources: map[string]v1.ResourceRequirements{"app": requirements("200m", "128Mi")}},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cp, err := NewCheckpointV2(&v1.PodList{Items: tt.pods})
			require.NoError(t, err)
			file, err := cp.MarshalCheckpoint()
			require.NoError(t, err)

			requireNoDiff(t, tt.want, readLegacyFile(t, file), "entries read by a kubelet that predates V2")
		})
	}
}

// A kubelet that predates V2 rewrites the file in its own format, which drops the pod list. The next
// upgrade then has to rebuild the pods from the entries alone.
func TestCheckpointV2_RollForwardAfterRollback(t *testing.T) {
	// This kubelet writes a V2 checkpoint.
	cp, err := NewCheckpointV2(&v1.PodList{Items: []v1.Pod{podWithSpec("uid-1", v1.PodSpec{
		InitContainers: []v1.Container{container("init", "100m", "64Mi")},
		Containers:     []v1.Container{container("app", "250m", "256Mi")},
		Resources:      ptr.To(requirements("500m", "512Mi")),
		Volumes:        []v1.Volume{emptyDirVolume("mem", v1.StorageMediumMemory, quantity("64Mi"))},
	})}})
	require.NoError(t, err)
	v2File, err := cp.MarshalCheckpoint()
	require.NoError(t, err)

	// After a rollback, the older kubelet reads the entries and, sooner or later, rewrites the file from them.
	v1File := writeLegacyFile(t, readLegacyFile(t, v2File))

	// After the next upgrade, this kubelet finds a V1 checkpoint again.
	restored := &Checkpoint{}
	require.NoError(t, restored.UnmarshalCheckpoint(v1File))
	require.NoError(t, restored.VerifyChecksum())
	got, migrated, err := restored.GetPodList()
	require.NoError(t, err)
	assert.True(t, migrated, "a V1 checkpoint must be reported as migrated, so that it gets rewritten as V2")

	// V1 recorded no container types and no volume medium, so the init container is now a regular one.
	want := []v1.Pod{podWithSpec("uid-1", v1.PodSpec{
		Containers: []v1.Container{
			container("app", "250m", "256Mi"),
			container("init", "100m", "64Mi"),
		},
		Resources: ptr.To(requirements("500m", "512Mi")),
		Volumes:   []v1.Volume{emptyDirVolume("mem", v1.StorageMediumDefault, quantity("64Mi"))},
	})}
	requireNoDiff(t, want, got.Items, "pods migrated after a rollback")
}

// V1 checkpoints carry no version and nothing but resources, so the pods are rebuilt from those.
func TestCheckpointV2_MigratesV1(t *testing.T) {
	tests := []struct {
		name    string
		entries PodResourceInfoMap
		want    []v1.Pod
	}{
		{name: "no pods"},
		{
			name: "containers are sorted by name",
			entries: PodResourceInfoMap{"uid-1": {ContainerResources: map[string]v1.ResourceRequirements{
				"zeta":  requirements("200m", "128Mi"),
				"alpha": requirements("100m", "64Mi"),
				"mid":   requirements("150m", "96Mi"),
			}}},
			want: []v1.Pod{podWithSpec("uid-1", v1.PodSpec{Containers: []v1.Container{
				container("alpha", "100m", "64Mi"),
				container("mid", "150m", "96Mi"),
				container("zeta", "200m", "128Mi"),
			}})},
		},
		{
			name:    "pod-level resources",
			entries: PodResourceInfoMap{"uid-1": {PodLevelResources: ptr.To(requirements("500m", "512Mi"))}},
			want:    []v1.Pod{podWithSpec("uid-1", v1.PodSpec{Resources: ptr.To(requirements("500m", "512Mi"))})},
		},
		{
			// V1 did not record the medium.
			name: "emptyDir limits become volumes with the default medium, sorted by name",
			entries: PodResourceInfoMap{"uid-1": {EmptyDirVolumeLimits: map[string]*resource.Quantity{
				"vol-z": quantity("2Mi"),
				"vol-a": quantity("1Mi"),
				"vol-m": quantity("3Mi"),
			}}},
			want: []v1.Pod{podWithSpec("uid-1", v1.PodSpec{Volumes: []v1.Volume{
				emptyDirVolume("vol-a", v1.StorageMediumDefault, quantity("1Mi")),
				emptyDirVolume("vol-m", v1.StorageMediumDefault, quantity("3Mi")),
				emptyDirVolume("vol-z", v1.StorageMediumDefault, quantity("2Mi")),
			}})},
		},
		{
			name:    "a nil emptyDir limit becomes a volume without a size limit",
			entries: PodResourceInfoMap{"uid-1": {EmptyDirVolumeLimits: map[string]*resource.Quantity{"vol": nil}}},
			want: []v1.Pod{podWithSpec("uid-1", v1.PodSpec{Volumes: []v1.Volume{
				emptyDirVolume("vol", v1.StorageMediumDefault, nil),
			}})},
		},
		{
			name: "pods are sorted by UID",
			entries: PodResourceInfoMap{
				"uid-3": {ContainerResources: map[string]v1.ResourceRequirements{"app": requirements("300m", "192Mi")}},
				"uid-1": {ContainerResources: map[string]v1.ResourceRequirements{"app": requirements("100m", "64Mi")}},
				"uid-2": {ContainerResources: map[string]v1.ResourceRequirements{"app": requirements("200m", "128Mi")}},
			},
			want: []v1.Pod{
				podWithSpec("uid-1", v1.PodSpec{Containers: []v1.Container{container("app", "100m", "64Mi")}}),
				podWithSpec("uid-2", v1.PodSpec{Containers: []v1.Container{container("app", "200m", "128Mi")}}),
				podWithSpec("uid-3", v1.PodSpec{Containers: []v1.Container{container("app", "300m", "192Mi")}}),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cp := &Checkpoint{}
			require.NoError(t, cp.UnmarshalCheckpoint(writeLegacyFile(t, tt.entries)))
			require.NoError(t, cp.VerifyChecksum())

			// Map iteration order is randomized on every iteration, so an unsorted result would only
			// differ some of the time.
			for range 20 {
				got, migrated, err := cp.GetPodList()
				require.NoError(t, err)
				require.True(t, migrated, "a V1 checkpoint must be reported as migrated, so that it gets rewritten as V2")
				requireNoDiff(t, tt.want, got.Items, "migrated pods")
			}
		})
	}
}

// The legacy entries in a V2 checkpoint exist for rollbacks only. Reading them instead of the pod
// list would lose everything the entries cannot express.
func TestCheckpointV2_PrefersPodListOverEntries(t *testing.T) {
	podList := &v1.PodList{Items: []v1.Pod{podWithSpec("from-pod-list", v1.PodSpec{})}}
	podListProto, err := podList.Marshal()
	require.NoError(t, err)
	data, err := json.Marshal(CheckpointData{
		PodListProto: podListProto,
		Entries:      PodResourceInfoMap{"from-entries": {}},
	})
	require.NoError(t, err)
	cp := &Checkpoint{Version: checkpointVersionV2, Data: string(data), Checksum: checksum.New(string(data))}

	got, migrated, err := cp.GetPodList()
	require.NoError(t, err)
	assert.False(t, migrated, "a V2 checkpoint must not be treated as V1")
	requireNoDiff(t, podList.Items, got.Items, "pods of a checkpoint whose entries disagree with its pod list")
}

// A V2 checkpoint without a pod list reads as having no pods, like a V1 checkpoint without entries.
// Failing instead would keep the kubelet from starting until an operator deletes the file.
func TestCheckpointV2_MissingPodList(t *testing.T) {
	tests := []struct {
		name string
		data string
	}{
		{name: "no podList key", data: `{}`},
		{name: "null podList", data: `{"podList":null}`},
		{name: "empty podList", data: `{"podList":""}`},
		// The legacy entries exist for rollbacks only, so they are no substitute for the pod list.
		{name: "entries but no podList", data: `{"entries":{"uid-1":{}}}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cp := &Checkpoint{Version: checkpointVersionV2, Data: tt.data, Checksum: checksum.New(tt.data)}

			got, migrated, err := cp.GetPodList()
			require.NoError(t, err)
			assert.False(t, migrated, "a V2 checkpoint must not be treated as V1")
			assert.Empty(t, got.Items)
		})
	}
}

// A checkpoint of an unknown version, such as one written by a newer kubelet before a downgrade, must
// not be read as V1: it would be misread as an empty state, and the next write would overwrite it.
func TestCheckpointV2_UnknownVersion(t *testing.T) {
	// "v1" is unknown too: V1 checkpoints carry no version at all.
	for _, version := range []string{"v3", "V2", "v1", "garbage"} {
		t.Run(version, func(t *testing.T) {
			// The data is well-formed for both known versions, so only the version can be at fault.
			cp := &Checkpoint{Version: version, Data: "{}", Checksum: checksum.New("{}")}

			got, migrated, err := cp.GetPodList()
			require.ErrorContains(t, err, fmt.Sprintf("unsupported checkpoint version %q", version))
			assert.Nil(t, got)
			assert.False(t, migrated)
		})
	}
}

// A payload that is present but cannot be decoded is an error, as it is for V1 checkpoints.
func TestCheckpointV2_CorruptPayload(t *testing.T) {
	// A length-delimited field that claims 5 bytes but has 1.
	truncatedPodList, err := json.Marshal(CheckpointData{PodListProto: []byte{0x0a, 0x05, 0x01}})
	require.NoError(t, err)

	tests := []struct {
		name    string
		version string
		data    string
		wantErr string
	}{
		{
			name:    "V2 data is not JSON",
			version: checkpointVersionV2,
			data:    "{not json",
			wantErr: "failed to unmarshal V2 checkpoint data",
		},
		{
			name:    "V2 pod list is not base64",
			version: checkpointVersionV2,
			data:    `{"podList":"not base64!"}`,
			wantErr: "failed to unmarshal V2 checkpoint data",
		},
		{
			name:    "V2 pod list is not a protobuf PodList",
			version: checkpointVersionV2,
			data:    string(truncatedPodList),
			wantErr: "failed to unmarshal protobuf PodList",
		},
		{
			name:    "V1 data is not JSON",
			data:    "{not json",
			wantErr: "failed to migrate legacy V1 checkpoint",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cp := &Checkpoint{Version: tt.version, Data: tt.data, Checksum: checksum.New(tt.data)}

			got, migrated, err := cp.GetPodList()
			require.ErrorContains(t, err, tt.wantErr)
			assert.Nil(t, got)
			assert.False(t, migrated)
		})
	}
}

// legacyCheckpointFile is how a kubelet that predates V2 sees a checkpoint file: it has no notion of
// a version, and expects Data to hold a PodResourceCheckpointInfo.
type legacyCheckpointFile struct {
	Data     string            `json:"data"`
	Checksum checksum.Checksum `json:"checksum"`
}

// readLegacyFile returns the entries that a kubelet that predates V2 restores from the file. Like
// that kubelet, it ignores everything but the entries, and it rejects a wrong checksum.
func readLegacyFile(t *testing.T, file []byte) PodResourceInfoMap {
	t.Helper()
	var legacy legacyCheckpointFile
	require.NoError(t, json.Unmarshal(file, &legacy))
	require.NoError(t, legacy.Checksum.Verify(legacy.Data), "a kubelet that predates V2 would reject this checkpoint")
	var payload PodResourceCheckpointInfo
	require.NoError(t, json.Unmarshal([]byte(legacy.Data), &payload))
	return payload.Entries
}

// writeLegacyFile returns the file that a kubelet that predates V2 writes for the entries.
func writeLegacyFile(t *testing.T, entries PodResourceInfoMap) []byte {
	t.Helper()
	data, err := json.Marshal(PodResourceCheckpointInfo{Entries: entries})
	require.NoError(t, err)
	file, err := json.Marshal(legacyCheckpointFile{Data: string(data), Checksum: checksum.New(string(data))})
	require.NoError(t, err)
	return file
}

// requireNoDiff fails the test with a diff if want and got differ. Quantities compare by value,
// because round trips through JSON and protobuf change their cached string form, and a nil slice or
// map equals an empty one.
func requireNoDiff(t *testing.T, want, got any, what string) {
	t.Helper()
	if diff := cmp.Diff(want, got, cmpopts.EquateEmpty()); diff != "" {
		t.Fatalf("unexpected %s (-want +got):\n%s", what, diff)
	}
}

// newFullPod returns a pod with every kind of container, pod-level resources, memory-backed volumes
// with and without a size limit, and enough labels and annotations for a nondeterministic encoding
// of maps to show up as differing bytes.
func newFullPod(uid types.UID) v1.Pod {
	return v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:         uid,
			Name:        "pod-" + string(uid),
			Namespace:   "ns",
			Labels:      map[string]string{"a": "1", "b": "2", "c": "3", "d": "4"},
			Annotations: map[string]string{"w": "1", "x": "2", "y": "3", "z": "4"},
		},
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{container("init", "100m", "64Mi")},
			Containers: []v1.Container{
				container("app", "250m", "256Mi"),
				container("sidecar", "50m", "32Mi"),
			},
			EphemeralContainers: []v1.EphemeralContainer{ephemeralContainer("debug", "10m", "8Mi")},
			Resources:           ptr.To(requirements("500m", "512Mi")),
			Volumes: []v1.Volume{
				emptyDirVolume("mem-a", v1.StorageMediumMemory, quantity("64Mi")),
				emptyDirVolume("mem-b", v1.StorageMediumMemory, quantity("128Mi")),
				emptyDirVolume("mem-unlimited", v1.StorageMediumMemory, nil),
			},
		},
	}
}

func podWithSpec(uid types.UID, spec v1.PodSpec) v1.Pod {
	return v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: uid}, Spec: spec}
}

func container(name, cpu, memory string) v1.Container {
	return v1.Container{Name: name, Resources: requirements(cpu, memory)}
}

func ephemeralContainer(name, cpu, memory string) v1.EphemeralContainer {
	return v1.EphemeralContainer{
		EphemeralContainerCommon: v1.EphemeralContainerCommon{Name: name, Resources: requirements(cpu, memory)},
	}
}

func emptyDirVolume(name string, medium v1.StorageMedium, limit *resource.Quantity) v1.Volume {
	return v1.Volume{
		Name:         name,
		VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{Medium: medium, SizeLimit: limit}},
	}
}

// requirements returns requests and limits that are both the given cpu and memory.
func requirements(cpu, memory string) v1.ResourceRequirements {
	list := func() v1.ResourceList {
		return v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse(cpu),
			v1.ResourceMemory: resource.MustParse(memory),
		}
	}
	return v1.ResourceRequirements{Requests: list(), Limits: list()}
}

func quantity(s string) *resource.Quantity {
	q := resource.MustParse(s)
	return &q
}
