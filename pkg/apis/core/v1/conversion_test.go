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

package v1_test

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"math/rand"
	"net/url"
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metafuzzer "k8s.io/apimachinery/pkg/apis/meta/fuzzer"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/diff"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	apps "k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/core"
	corefuzzer "k8s.io/kubernetes/pkg/apis/core/fuzzer"
	corev1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/yaml"

	// ensure types are installed
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	// ensure types are installed corereplicationcontroller<->replicaset conversions
	_ "k8s.io/kubernetes/pkg/apis/apps/install"
)

//go:embed testdata/exemplar_pod.yaml
var benchmarkExemplarPodYAML []byte

const benchmarkSeed = 100

func TestPodLogOptions(t *testing.T) {
	sinceSeconds := int64(1)
	sinceTime := metav1.NewTime(time.Date(2000, 1, 1, 12, 34, 56, 0, time.UTC).Local())
	tailLines := int64(2)
	limitBytes := int64(3)
	v1StreamStderr := v1.LogStreamStderr
	coreStreamStderr := core.LogStreamStderr

	versionedLogOptions := &v1.PodLogOptions{
		Container:    "mycontainer",
		Follow:       true,
		Previous:     true,
		SinceSeconds: &sinceSeconds,
		SinceTime:    &sinceTime,
		Timestamps:   true,
		TailLines:    &tailLines,
		LimitBytes:   &limitBytes,
		Stream:       &v1StreamStderr,
	}
	unversionedLogOptions := &core.PodLogOptions{
		Container:    "mycontainer",
		Follow:       true,
		Previous:     true,
		SinceSeconds: &sinceSeconds,
		SinceTime:    &sinceTime,
		Timestamps:   true,
		TailLines:    &tailLines,
		LimitBytes:   &limitBytes,
		Stream:       &coreStreamStderr,
	}
	expectedParameters := url.Values{
		"container":    {"mycontainer"},
		"follow":       {"true"},
		"previous":     {"true"},
		"sinceSeconds": {"1"},
		"sinceTime":    {"2000-01-01T12:34:56Z"},
		"timestamps":   {"true"},
		"tailLines":    {"2"},
		"limitBytes":   {"3"},
		"stream":       {"Stderr"},
	}

	codec := runtime.NewParameterCodec(legacyscheme.Scheme)

	// unversioned -> query params
	{
		actualParameters, err := codec.EncodeParameters(unversionedLogOptions, v1.SchemeGroupVersion)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(actualParameters, expectedParameters) {
			t.Fatalf("Expected\n%#v\ngot\n%#v", expectedParameters, actualParameters)
		}
	}

	// versioned -> query params
	{
		actualParameters, err := codec.EncodeParameters(versionedLogOptions, v1.SchemeGroupVersion)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(actualParameters, expectedParameters) {
			t.Fatalf("Expected\n%#v\ngot\n%#v", expectedParameters, actualParameters)
		}
	}

	// query params -> versioned
	{
		convertedLogOptions := &v1.PodLogOptions{}
		err := codec.DecodeParameters(expectedParameters, v1.SchemeGroupVersion, convertedLogOptions)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(convertedLogOptions, versionedLogOptions) {
			t.Fatalf("Unexpected deserialization:\n%s", diff.ObjectGoPrintSideBySide(versionedLogOptions, convertedLogOptions))
		}
	}

	// query params -> unversioned
	{
		convertedLogOptions := &core.PodLogOptions{}
		err := codec.DecodeParameters(expectedParameters, v1.SchemeGroupVersion, convertedLogOptions)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(convertedLogOptions, unversionedLogOptions) {
			t.Fatalf("Unexpected deserialization:\n%s", diff.ObjectGoPrintSideBySide(unversionedLogOptions, convertedLogOptions))
		}
	}
}

// TestPodSpecConversion tests that ServiceAccountName and its deprecated alias
// are preserved verbatim by conversion in both directions (keeping them in
// sync is handled by defaulting and the pod registry strategy, not conversion).
func TestPodSpecConversion(t *testing.T) {
	name, other := "foo", "bar"

	i := &core.PodSpec{
		ServiceAccountName:       name,
		DeprecatedServiceAccount: other,
	}
	v := v1.PodSpec{}
	if err := legacyscheme.Scheme.Convert(i, &v, nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v.ServiceAccountName != name {
		t.Fatalf("want v1.ServiceAccountName %q, got %q", name, v.ServiceAccountName)
	}
	if v.DeprecatedServiceAccount != other {
		t.Fatalf("want v1.DeprecatedServiceAccount %q, got %q", other, v.DeprecatedServiceAccount)
	}

	got := core.PodSpec{}
	if err := legacyscheme.Scheme.Convert(&v, &got, nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.ServiceAccountName != name {
		t.Fatalf("want core.ServiceAccountName %q, got %q", name, got.ServiceAccountName)
	}
	if got.DeprecatedServiceAccount != other { //nolint:staticcheck // SA1019 DeprecatedServiceAccount must be tested for backward compatibility
		t.Fatalf("want core.DeprecatedServiceAccount %q, got %q", other, got.DeprecatedServiceAccount) //nolint:staticcheck // SA1019 DeprecatedServiceAccount must be tested for backward compatibility
	}
}

func TestPodSpecHostNamespaceSecurityContextRoundTrip(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(spec *v1.PodSpec)
	}{
		{"none", func(spec *v1.PodSpec) {}},
		{"hostNetwork", func(spec *v1.PodSpec) { spec.HostNetwork = true }},
		{"hostPID", func(spec *v1.PodSpec) { spec.HostPID = true }},
		{"hostIPC", func(spec *v1.PodSpec) { spec.HostIPC = true }},
		{"hostUsers", func(spec *v1.PodSpec) { spec.HostUsers = new(true) }},
		{"hostUsersFalse", func(spec *v1.PodSpec) { spec.HostUsers = new(false) }},
		{"shareProcessNamespace", func(spec *v1.PodSpec) { spec.ShareProcessNamespace = new(true) }},
		{"shareProcessNamespaceFalse", func(spec *v1.PodSpec) { spec.ShareProcessNamespace = new(false) }},
		{"all", func(spec *v1.PodSpec) {
			spec.HostNetwork = true
			spec.HostIPC = true
			spec.HostUsers = new(true)
			spec.ShareProcessNamespace = new(true)
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			in := &v1.Pod{}
			tc.mutate(&in.Spec)
			if in.Spec.SecurityContext != nil {
				t.Fatalf("securityContext must start nil")
			}
			legacyscheme.Scheme.Default(in)
			if in.Spec.SecurityContext == nil {
				t.Errorf("securityContext must default to non-nil")
			}

			// v1 -> internal
			internal := &core.Pod{}
			if err := legacyscheme.Scheme.Convert(in, internal, nil); err != nil {
				t.Fatal(err)
			}
			// compare across types
			if internal.Spec.HostNetwork != in.Spec.HostNetwork ||
				internal.Spec.HostPID != in.Spec.HostPID ||
				internal.Spec.HostIPC != in.Spec.HostIPC ||
				!reflect.DeepEqual(internal.Spec.HostUsers, in.Spec.HostUsers) ||
				!reflect.DeepEqual(internal.Spec.ShareProcessNamespace, in.Spec.ShareProcessNamespace) {
				t.Errorf("v1 -> internal:\n%s", cmp.Diff(in.Spec, internal.Spec))
			}

			// internal -> v1
			out := &v1.Pod{}
			if err := legacyscheme.Scheme.Convert(internal, out, nil); err != nil {
				t.Fatal(err)
			}
			if out.Spec.SecurityContext == nil {
				t.Errorf("securityContext must be an non-nil")
			}
			// compare round-tripped
			if !reflect.DeepEqual(in.Spec, out.Spec) {
				t.Errorf("internal -> v1:\n%s", cmp.Diff(in.Spec, out.Spec))
			}
		})
	}
}

func BenchmarkPodListConversion(b *testing.B) {
	versionedPod := loadExemplarPod()

	for _, tc := range []struct {
		name      string
		benchmark func(*testing.B, int, *v1.Pod)
	}{
		{name: "core-to-v1", benchmark: benchmarkConvertCorePodListToV1},
		{name: "v1-to-core", benchmark: benchmarkConvertV1PodListToCore},
	} {
		b.Run(tc.name, func(b *testing.B) {
			for _, size := range []int{1, 100, 1000, 5000, 10000} {
				b.Run(fmt.Sprintf("pods=%d", size), func(b *testing.B) {
					tc.benchmark(b, size, versionedPod)
				})
			}
		})
	}
}

func benchmarkConvertCorePodListToV1(b *testing.B, size int, exemplar *v1.Pod) {
	in := benchmarkCorePodList(b, benchmarkVersionedPodList(exemplar, size))

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			out := &v1.PodList{}
			if err := legacyscheme.Scheme.Convert(in, out, nil); err != nil {
				panic(fmt.Sprintf("unexpected conversion error: %v", err))
			}
			if len(out.Items) != size {
				panic(fmt.Sprintf("unexpected converted item count: got %d, want %d", len(out.Items), size))
			}
		}
	})
}

func benchmarkConvertV1PodListToCore(b *testing.B, size int, exemplar *v1.Pod) {
	in := benchmarkVersionedPodList(exemplar, size)

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			out := &core.PodList{}
			if err := legacyscheme.Scheme.Convert(in, out, nil); err != nil {
				panic(fmt.Sprintf("unexpected conversion error: %v", err))
			}
			if len(out.Items) != size {
				panic(fmt.Sprintf("unexpected converted item count: got %d, want %d", len(out.Items), size))
			}
		}
	})
}

func benchmarkCorePodList(b *testing.B, in *v1.PodList) *core.PodList {
	b.Helper()
	out := &core.PodList{}
	if err := legacyscheme.Scheme.Convert(in, out, nil); err != nil {
		b.Fatalf("unexpected setup conversion error: %v", err)
	}
	return out
}

func benchmarkVersionedPodList(exemplar *v1.Pod, size int) *v1.PodList {
	utilrand.Seed(benchmarkSeed)
	nodeNames := make([]string, 25)
	for i := range nodeNames {
		nodeNames[i] = utilrand.String(10)
	}
	items := make([]v1.Pod, size)
	for i := range items {
		pod := exemplar.DeepCopy()
		randomizePod(pod, utilrand.String(10), nodeNames[utilrand.Intn(len(nodeNames))])
		items[i] = *pod
	}
	return &v1.PodList{
		ListMeta: metav1.ListMeta{ResourceVersion: "1000"},
		Items:    items,
	}
}

func randomizePod(pod *v1.Pod, ns string, nodeName string) {
	pod.Namespace = ns
	pod.Name = pod.GenerateName + utilrand.String(10)
	pod.UID = types.UID(utilrand.String(36))
	pod.ResourceVersion = ""
	pod.Spec.NodeName = nodeName
}

func loadExemplarPod() *v1.Pod {
	if len(benchmarkExemplarPodYAML) == 0 {
		panic("exemplar pod empty")
	}
	pod := &v1.Pod{}
	if err := yaml.Unmarshal(benchmarkExemplarPodYAML, pod); err != nil {
		panic(fmt.Sprintf("decode exemplar pod: %v", err))
	}
	return pod
}

func TestResourceListConversion(t *testing.T) {
	bigMilliQuantity := resource.NewQuantity(resource.MaxMilliValue, resource.DecimalSI)
	bigMilliQuantity.Add(resource.MustParse("12345m"))

	tests := []struct {
		input    v1.ResourceList
		expected core.ResourceList
	}{
		{ // No changes necessary.
			input: v1.ResourceList{
				v1.ResourceMemory:  resource.MustParse("30M"),
				v1.ResourceCPU:     resource.MustParse("100m"),
				v1.ResourceStorage: resource.MustParse("1G"),
			},
			expected: core.ResourceList{
				core.ResourceMemory:  resource.MustParse("30M"),
				core.ResourceCPU:     resource.MustParse("100m"),
				core.ResourceStorage: resource.MustParse("1G"),
			},
		},
		{ // Nano-scale values should be rounded up to milli-scale.
			input: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("3.000023m"),
				v1.ResourceMemory: resource.MustParse("500.000050m"),
			},
			expected: core.ResourceList{
				core.ResourceCPU:    resource.MustParse("4m"),
				core.ResourceMemory: resource.MustParse("501m"),
			},
		},
		{ // Large values should still be accurate.
			input: v1.ResourceList{
				v1.ResourceCPU:     bigMilliQuantity.DeepCopy(),
				v1.ResourceStorage: bigMilliQuantity.DeepCopy(),
			},
			expected: core.ResourceList{
				core.ResourceCPU:     bigMilliQuantity.DeepCopy(),
				core.ResourceStorage: bigMilliQuantity.DeepCopy(),
			},
		},
	}

	for i, test := range tests {
		output := core.ResourceList{}

		// defaulting is a separate step from conversion that is applied when reading from the API or from etcd.
		// perform that step explicitly.
		corev1.SetDefaults_ResourceList(&test.input)

		err := legacyscheme.Scheme.Convert(&test.input, &output, nil)
		if err != nil {
			t.Fatalf("unexpected error for case %d: %v", i, err)
		}
		if !apiequality.Semantic.DeepEqual(test.expected, output) {
			t.Errorf("unexpected conversion for case %d: Expected\n%+v;\nGot\n%+v", i, test.expected, output)
		}
	}
}

func TestReplicationControllerConversion(t *testing.T) {
	// If we start with a RC, we should always have round-trip fidelity.
	inputs := []*v1.ReplicationController{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "name",
				Namespace: "namespace",
			},
			Spec: v1.ReplicationControllerSpec{
				Replicas:        ptr.To[int32](1),
				MinReadySeconds: 32,
				Selector:        map[string]string{"foo": "bar", "bar": "foo"},
				Template: &v1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Labels: map[string]string{"foo": "bar", "bar": "foo"},
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name:  "container",
								Image: "image",
							},
						},
					},
				},
			},
			Status: v1.ReplicationControllerStatus{
				Replicas:             1,
				FullyLabeledReplicas: 2,
				ReadyReplicas:        3,
				AvailableReplicas:    4,
				ObservedGeneration:   5,
				Conditions: []v1.ReplicationControllerCondition{
					{
						Type:               v1.ReplicationControllerReplicaFailure,
						Status:             v1.ConditionTrue,
						LastTransitionTime: metav1.NewTime(time.Unix(123456789, 0)),
						Reason:             "Reason",
						Message:            "Message",
					},
				},
			},
		},
	}

	// Add some fuzzed RCs.
	apiObjectFuzzer := fuzzer.FuzzerFor(fuzzer.MergeFuzzerFuncs(metafuzzer.Funcs, corefuzzer.Funcs), rand.NewSource(152), legacyscheme.Codecs)
	for i := 0; i < 100; i++ {
		rc := &v1.ReplicationController{}
		apiObjectFuzzer.Fill(rc)
		// Sometimes the fuzzer decides to leave Spec.Template nil.
		// We can't support that because Spec.Template is not a pointer in RS,
		// so it will round-trip as non-nil but empty.
		if rc.Spec.Template == nil {
			rc.Spec.Template = &v1.PodTemplateSpec{}
		}
		// Sometimes the fuzzer decides to insert an empty label key.
		// This doesn't round-trip properly because it's invalid.
		if rc.Spec.Selector != nil {
			delete(rc.Spec.Selector, "")
		}
		inputs = append(inputs, rc)
	}

	// Round-trip the input RCs before converting to RS.
	for i := range inputs {
		inputs[i] = roundTrip(t, inputs[i]).(*v1.ReplicationController)
	}

	for _, in := range inputs {
		rs := &apps.ReplicaSet{}
		// Use in.DeepCopy() to avoid sharing pointers with `in`.
		if err := corev1.Convert_v1_ReplicationController_To_apps_ReplicaSet(in.DeepCopy(), rs, nil); err != nil {
			t.Errorf("can't convert RC to RS: %v", err)
			continue
		}
		// Round-trip RS before converting back to RC.
		rs = roundTripRS(t, rs)
		out := &v1.ReplicationController{}
		if err := corev1.Convert_apps_ReplicaSet_To_v1_ReplicationController(rs, out, nil); err != nil {
			t.Errorf("can't convert RS to RC: %v", err)
			continue
		}
		if !apiequality.Semantic.DeepEqual(in, out) {
			instr, _ := json.MarshalIndent(in, "", "  ")
			outstr, _ := json.MarshalIndent(out, "", "  ")
			t.Errorf("RC-RS conversion round-trip failed:\nin:\n%s\nout:\n%s\ndiff:\n%s", instr, outstr, cmp.Diff(in, out))
		}
	}
}

func TestReplicaSetConversion(t *testing.T) {
	inputs := []*apps.ReplicaSet{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "name",
				Namespace: "namespace",
				Labels:    map[string]string{"foo": "bar", "bar": "foo"}, // labels have to be defined everywhere not to trigger RC defaulting
			},
			Spec: apps.ReplicaSetSpec{
				Replicas:        1,
				MinReadySeconds: 32,
				Selector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"foo": "bar", "bar": "foo"},
				},
				Template: core.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Labels: map[string]string{"foo": "bar", "bar": "foo"},
					},
					Spec: core.PodSpec{
						Containers: []core.Container{
							{
								Name:  "container",
								Image: "image",
							},
						},
					},
				},
			},
			Status: apps.ReplicaSetStatus{
				Replicas:             1,
				FullyLabeledReplicas: 2,
				ReadyReplicas:        3,
				AvailableReplicas:    4,
				TerminatingReplicas:  nil, // ReplicationController does not support .status.terminatingReplicas
				ObservedGeneration:   5,
				Conditions: []apps.ReplicaSetCondition{
					{
						Type:               apps.ReplicaSetReplicaFailure,
						Status:             core.ConditionTrue,
						LastTransitionTime: metav1.NewTime(time.Unix(123456789, 0)),
						Reason:             "Reason",
						Message:            "Message",
					},
				},
			},
		},
	}

	// Add some fuzzed ReplicaSets.
	apiObjectFuzzer := fuzzer.FuzzerFor(fuzzer.MergeFuzzerFuncs(metafuzzer.Funcs, corefuzzer.Funcs), rand.NewSource(152), legacyscheme.Codecs)
	for i := 0; i < 100; i++ {
		rs := &apps.ReplicaSet{}
		apiObjectFuzzer.Fill(rs)
		// if we have labels, they have to be set to satisfy RC defaulting
		if labels := rs.Spec.Template.Labels; len(labels) > 0 {
			rs.Labels = labels

			// forcefully set label selector, since the RC has only partial selector support (MatchLabels)
			rs.Spec.Selector = &metav1.LabelSelector{
				MatchLabels: labels,
			}
		} else {
			rs.Spec.Selector = nil
		}

		// ReplicationController does not support .status.terminatingReplicas
		if rs.Status.TerminatingReplicas != nil {
			rs.Status.TerminatingReplicas = nil
		}
		inputs = append(inputs, rs)
	}

	// Round-trip the input RSs before converting to RC.
	for i := range inputs {
		inputs[i] = roundTripRS(t, inputs[i])
	}

	for _, in := range inputs {
		rc := &v1.ReplicationController{}
		// Use in.DeepCopy() to avoid sharing pointers with `in`.
		if err := corev1.Convert_apps_ReplicaSet_To_v1_ReplicationController(in.DeepCopy(), rc, nil); err != nil {
			t.Errorf("can't convert RS to RC: %v", err)
			continue
		}
		// Round-trip RC before converting back to RS.
		rc = roundTrip(t, rc).(*v1.ReplicationController)
		out := &apps.ReplicaSet{}
		if err := corev1.Convert_v1_ReplicationController_To_apps_ReplicaSet(rc, out, nil); err != nil {
			t.Errorf("can't convert RC to RS: %v", err)
			continue
		}
		if !apiequality.Semantic.DeepEqual(in, out) {
			instr, _ := json.MarshalIndent(in, "", "  ")
			outstr, _ := json.MarshalIndent(out, "", "  ")
			t.Errorf("RS-RC conversion round-trip failed:\nin:\n%s\nout:\n%s\ndiff:\n%s", instr, outstr, cmp.Diff(in, out))
		}
	}
}

func roundTripRS(t *testing.T, rs *apps.ReplicaSet) *apps.ReplicaSet {
	codec := legacyscheme.Codecs.LegacyCodec(appsv1.SchemeGroupVersion)
	data, err := runtime.Encode(codec, rs)
	if err != nil {
		t.Errorf("%v\n %#v", err, rs)
		return nil
	}
	obj2, err := runtime.Decode(codec, data)
	if err != nil {
		t.Errorf("%v\nData: %s\nSource: %#v", err, string(data), rs)
		return nil
	}
	return obj2.(*apps.ReplicaSet)
}

func TestPodStatusConversion(t *testing.T) {
	in := core.PodStatus{
		PodIP: "1.1.2.1",
		PodIPs: []core.PodIP{
			{IP: "1.1.1.1"},
			{IP: "2000::"},
		},
	}
	v1Status := v1.PodStatus{}
	if err := corev1.Convert_core_PodStatus_To_v1_PodStatus(&in, &v1Status, nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v1Status.PodIP != in.PodIP {
		t.Errorf("expected v1 podIP %q, got %q", in.PodIP, v1Status.PodIP)
	}
	if len(v1Status.PodIPs) != len(in.PodIPs) || v1Status.PodIPs[0].IP != in.PodIPs[0].IP {
		t.Errorf("expected v1 podIPs to match, got %#v", v1Status.PodIPs)
	}

	roundTripped := core.PodStatus{}
	if err := corev1.Convert_v1_PodStatus_To_core_PodStatus(&v1Status, &roundTripped, nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if roundTripped.PodIP != in.PodIP || len(roundTripped.PodIPs) != len(in.PodIPs) {
		t.Errorf("expected round-tripped to match, got %#v", roundTripped)
	}
}

func Test_core_NodeSpec_to_v1_NodeSpec(t *testing.T) {
	// core to v1
	testInputs := []core.NodeSpec{
		{
			PodCIDRs: []string{"10.0.0.0/24", "10.0.1.0/24"},
		},
		{
			PodCIDRs: nil,
		},
		{
			PodCIDRs: []string{"10.0.0.0/24"},
		},
		{
			PodCIDRs: []string{"ace:cab:deca::/8"},
		},
		{
			PodCIDRs: []string{"10.0.0.0/24", "ace:cab:deca::/8"},
		},
		{
			PodCIDRs: []string{"ace:cab:deca::/8", "10.0.0.0/24"},
		},
	}

	for i, testInput := range testInputs {
		v1NodeSpec := v1.NodeSpec{}
		// convert
		if err := corev1.Convert_core_NodeSpec_To_v1_NodeSpec(&testInput, &v1NodeSpec, nil); nil != err {
			t.Errorf("%v: Convert core.NodeSpec to v1.NodeSpec failed with error %v", i, err.Error())
		}

		if len(testInput.PodCIDRs) == 0 {
			continue // no more work needed
		}

		// validate results
		if v1NodeSpec.PodCIDR != testInput.PodCIDRs[0] {
			t.Errorf("%v: Convert core.NodeSpec to v1.NodeSpec failed. Expected v1.PodCIDR=%v but found %v", i, testInput.PodCIDRs[0], v1NodeSpec.PodCIDR)
		}

		// match v1.PodIPs to core.PodIPs
		for idx := range testInput.PodCIDRs {
			if v1NodeSpec.PodCIDRs[idx] != testInput.PodCIDRs[idx] {
				t.Errorf("%v: Convert core.NodeSpec to v1.NodeSpec failed. Expected v1.NodeSpec[%v]=%v but found %v", i, idx, testInput.PodCIDRs[idx], v1NodeSpec.PodCIDRs[idx])
			}
		}
	}
}

func Test_v1_NodeSpec_to_core_NodeSpec(t *testing.T) {
	asymmetricInputs := []struct {
		name string
		in   v1.NodeSpec
		out  core.NodeSpec
	}{
		{
			name: "mismatched podCIDR",
			in: v1.NodeSpec{
				PodCIDR:  "10.0.0.0/24",
				PodCIDRs: []string{"10.0.1.0/24", "ace:cab:deca::/8"},
			},
			out: core.NodeSpec{
				PodCIDRs: []string{"10.0.0.0/24"},
			},
		},
		{
			name: "unset podCIDR",
			in: v1.NodeSpec{
				PodCIDR:  "",
				PodCIDRs: []string{"10.0.1.0/24", "ace:cab:deca::/8"},
			},
			out: core.NodeSpec{
				PodCIDRs: []string{"10.0.1.0/24", "ace:cab:deca::/8"},
			},
		},
		{
			name: "matching podCIDR",
			in: v1.NodeSpec{
				PodCIDR:  "10.0.1.0/24",
				PodCIDRs: []string{"10.0.1.0/24", "ace:cab:deca::/8"},
			},
			out: core.NodeSpec{
				PodCIDRs: []string{"10.0.1.0/24", "ace:cab:deca::/8"},
			},
		},
	}

	testInputs := []v1.NodeSpec{
		// cidr only - 4
		{
			PodCIDR: "10.0.1.0/24",
		},
		// cidr only - 6
		{
			PodCIDR: "ace:cab:deca::/8",
		},
		// Both are provided
		{
			PodCIDR:  "10.0.1.0/24",
			PodCIDRs: []string{"10.0.1.0/24", "ace:cab:deca::/8"},
		},
		// list only
		{
			PodCIDRs: []string{"10.0.1.0/24", "ace:cab:deca::/8"},
		},
		// Both are provided 4,6
		{
			PodCIDR:  "10.0.1.0/24",
			PodCIDRs: []string{"10.0.1.0/24", "ace:cab:deca::/8"},
		},
		// Both are provided 6,4
		{
			PodCIDR:  "ace:cab:deca::/8",
			PodCIDRs: []string{"ace:cab:deca::/8", "10.0.1.0/24"},
		},
		// list only 4,6
		{
			PodCIDRs: []string{"10.0.1.0/24", "ace:cab:deca::/8"},
		},
		// list only 6,4
		{
			PodCIDRs: []string{"ace:cab:deca::/8", "10.0.1.0/24"},
		},
		// no cidr and no cidrs
		{
			PodCIDR:  "",
			PodCIDRs: nil,
		},
	}

	// run asymmetric cases
	for _, tc := range asymmetricInputs {
		testInput := tc.in

		coreNodeSpec := core.NodeSpec{}
		// convert..
		if err := corev1.Convert_v1_NodeSpec_To_core_NodeSpec(&testInput, &coreNodeSpec, nil); err != nil {
			t.Errorf("%s: Convert v1.NodeSpec to core.NodeSpec failed with error:%v for input %+v", tc.name, err.Error(), testInput)
		}
		if !reflect.DeepEqual(coreNodeSpec, tc.out) {
			t.Errorf("%s: expected %#v, got %#v", tc.name, tc.out.PodCIDRs, coreNodeSpec.PodCIDRs)
		}
	}

	for i, testInput := range testInputs {
		coreNodeSpec := core.NodeSpec{}
		if err := corev1.Convert_v1_NodeSpec_To_core_NodeSpec(&testInput, &coreNodeSpec, nil); err != nil {
			t.Errorf("%v:Convert v1.NodeSpec to core.NodeSpec failed with error:%v", i, err.Error())
		}
		if len(testInput.PodCIDRs) == 0 && len(testInput.PodCIDR) == 0 {
			continue // no more work needed
		}
		if len(testInput.PodCIDR) > 0 && coreNodeSpec.PodCIDRs[0] != testInput.PodCIDR {
			t.Errorf("%v:Convert v1.NodeSpec to core.NodeSpec failed. expected coreNodeSpec.PodCIDRs[0]=%v found %v", i, testInput.PodCIDR, coreNodeSpec.PodCIDRs[0])
		}
		// match ip list
		for idx := range testInput.PodCIDRs {
			if coreNodeSpec.PodCIDRs[idx] != testInput.PodCIDRs[idx] {
				t.Errorf("%v:Convert v1.NodeSpec to core.NodeSpec failed core.PodCIDRs[%v]=%v expected %v", i, idx, coreNodeSpec.PodCIDRs[idx], testInput.PodCIDRs[idx])
			}
		}
	}
}

func TestConvert_v1_Pod_To_core_Pod(t *testing.T) {
	type args struct {
		in  *v1.Pod
		out *core.Pod
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
		wantOut *core.Pod
	}{
		{
			args: args{
				in: &v1.Pod{
					Spec: v1.PodSpec{
						TerminationGracePeriodSeconds: ptr.To[int64](-1),
					},
				},
				out: &core.Pod{},
			},
			wantOut: &core.Pod{
				Spec: core.PodSpec{
					// This is impossible in practice because defaulting will coherce a negative value to 1.
					// We test it here to ensure faithful conversion.
					TerminationGracePeriodSeconds: ptr.To[int64](-1),
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := corev1.Convert_v1_Pod_To_core_Pod(tt.args.in, tt.args.out, nil); (err != nil) != tt.wantErr {
				t.Errorf("Convert_v1_Pod_To_core_Pod() error = %v, wantErr %v", err, tt.wantErr)
			}
			if diff := cmp.Diff(tt.args.out, tt.wantOut); diff != "" {
				t.Errorf("Convert_v1_Pod_To_core_Pod() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestConvert_core_Pod_To_v1_Pod(t *testing.T) {
	type args struct {
		in  *core.Pod
		out *v1.Pod
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
		wantOut *v1.Pod
	}{
		{
			args: args{
				in: &core.Pod{
					Spec: core.PodSpec{
						TerminationGracePeriodSeconds: ptr.To[int64](-1),
					},
				},
				out: &v1.Pod{},
			},
			wantOut: &v1.Pod{
				Spec: v1.PodSpec{
					// This is impossible in practice because defaulting will coherce a negative value to 1.
					// We test it here to ensure faithful conversion.
					TerminationGracePeriodSeconds: ptr.To[int64](-1),
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := corev1.Convert_core_Pod_To_v1_Pod(tt.args.in, tt.args.out, nil); (err != nil) != tt.wantErr {
				t.Errorf("Convert_core_Pod_To_v1_Pod() error = %v, wantErr %v", err, tt.wantErr)
			}
			if diff := cmp.Diff(tt.args.out, tt.wantOut); diff != "" {
				t.Errorf("Convert_core_Pod_To_v1_Pod() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestPodMemoryIdenticalConversion ensures the internal and v1 Pod types
// remain memory-identical. These types must be kept memory-identical
// for performance reasons.
func TestPodMemoryIdenticalConversion(t *testing.T) {
	f := fuzzer.FuzzerFor(fuzzer.MergeFuzzerFuncs(metafuzzer.Funcs, corefuzzer.Funcs), rand.NewSource(1), legacyscheme.Codecs).NilChance(0).NumElements(1, 1)
	t.Run("v1 to internal", func(t *testing.T) {
		in := &v1.Pod{}
		f.Fill(in)
		out := &core.Pod{}
		if err := legacyscheme.Scheme.Convert(in, out, nil); err != nil {
			t.Fatalf("conversion failed: %v", err)
		}
		assertMemoryIdentical(t, "Pod", reflect.ValueOf(in).Elem(), reflect.ValueOf(out).Elem())
	})
	t.Run("internal to v1", func(t *testing.T) {
		in := &core.Pod{}
		f.Fill(in)
		out := &v1.Pod{}
		if err := legacyscheme.Scheme.Convert(in, out, nil); err != nil {
			t.Fatalf("conversion failed: %v", err)
		}
		assertMemoryIdentical(t, "Pod", reflect.ValueOf(in).Elem(), reflect.ValueOf(out).Elem())
	})
}

// TestPodTemplateSpecMemoryIdenticalConversion ensures the internal and v1 PodTemplateSpec types are identical
func TestPodTemplateSpecMemoryIdenticalConversion(t *testing.T) {
	scheme := legacyscheme.Scheme
	out := &core.PodTemplateSpec{}
	in := &v1.PodTemplateSpec{}
	if err := scheme.Convert(in, out, nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertMemoryIdentical(t, "PodTemplateSpec", reflect.ValueOf(in).Elem(), reflect.ValueOf(out).Elem())

	out2 := &v1.PodTemplateSpec{}
	if err := scheme.Convert(out, out2, nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertMemoryIdentical(t, "PodTemplateSpec", reflect.ValueOf(in).Elem(), reflect.ValueOf(out2).Elem())
}

// TestServiceSpecMemoryIdenticalConversion ensures the internal and v1 ServiceSpec types are identical
func TestServiceSpecMemoryIdenticalConversion(t *testing.T) {
	scheme := legacyscheme.Scheme
	out := &core.ServiceSpec{}
	in := &v1.ServiceSpec{}
	if err := scheme.Convert(in, out, nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertMemoryIdentical(t, "ServiceSpec", reflect.ValueOf(in).Elem(), reflect.ValueOf(out).Elem())

	out2 := &v1.ServiceSpec{}
	if err := scheme.Convert(out, out2, nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertMemoryIdentical(t, "ServiceSpec", reflect.ValueOf(in).Elem(), reflect.ValueOf(out2).Elem())
}

// TestPersistentVolumeSourceMemoryIdenticalConversion ensures the internal and v1 PersistentVolumeSource types are identical
func TestPersistentVolumeSourceMemoryIdenticalConversion(t *testing.T) {
	scheme := legacyscheme.Scheme
	out := &core.PersistentVolumeSource{}
	in := &v1.PersistentVolumeSource{}
	if err := scheme.Convert(in, out, nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertMemoryIdentical(t, "PersistentVolumeSource", reflect.ValueOf(in).Elem(), reflect.ValueOf(out).Elem())

	out2 := &v1.PersistentVolumeSource{}
	if err := scheme.Convert(out, out2, nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertMemoryIdentical(t, "PersistentVolumeSource", reflect.ValueOf(in).Elem(), reflect.ValueOf(out2).Elem())
}

// TestReplicationControllerMemoryIdenticalConversion ensures the internal and v1 ReplicationController types are identical
func TestReplicationControllerMemoryIdenticalConversion(t *testing.T) {
	scheme := legacyscheme.Scheme
	out := &core.ReplicationController{}
	in := &v1.ReplicationController{}
	if err := scheme.Convert(in, out, nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertMemoryIdentical(t, "ReplicationController", reflect.ValueOf(in).Elem(), reflect.ValueOf(out).Elem())

	out2 := &v1.ReplicationController{}
	if err := scheme.Convert(out, out2, nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertMemoryIdentical(t, "ReplicationController", reflect.ValueOf(in).Elem(), reflect.ValueOf(out2).Elem())
}

func assertMemoryIdentical(t *testing.T, path string, a, b reflect.Value) {
	t.Helper()
	if a.Kind() != b.Kind() {
		t.Errorf("%s: unexpected kind mismatch: %s, %s", path, a.Kind(), b.Kind())
		return
	}
	switch a.Kind() {
	case reflect.Struct: // allow for copied structs since conversion copies status/spec today
		if a.Type().Size() != b.Type().Size() {
			t.Errorf("%s: unexpected struct size mismatch: %d, %d", path, a.Type().Size(), b.Type().Size())
			return
		}
		if a.NumField() != b.NumField() {
			t.Errorf("%s: unexpected field count mismatch: %d, %d", path, a.NumField(), b.NumField())
			return
		}
		for i := 0; i < a.NumField(); i++ {
			aTypeField := a.Type().Field(i)
			bTypeField := b.Type().Field(i)
			if aTypeField.Name != bTypeField.Name {
				t.Errorf("%s: unexpected field name mismatch: %s, %s", path, aTypeField.Name, bTypeField.Name)
			}
			if aTypeField.Offset != bTypeField.Offset {
				t.Errorf("%s.%s: unexpected field offset mismatch: %d, %d", path, aTypeField.Name, aTypeField.Offset, bTypeField.Offset)
			}
			assertMemoryIdentical(t, path+"."+aTypeField.Name, a.Field(i), b.Field(i))
		}
	case reflect.Pointer, reflect.Map, reflect.Slice:
		if a.IsNil() != b.IsNil() {
			t.Errorf("%s: nilable pointer mismatch: %v, %v", path, !a.IsNil(), !b.IsNil())
			return
		}
		if !a.IsNil() && a.UnsafePointer() != b.UnsafePointer() {
			t.Errorf("%s: nilable type was unexpectedly copied", path)
		}
	case reflect.Bool, reflect.Int, reflect.Int32, reflect.Int64, reflect.Uint, reflect.Uint32, reflect.Uint64,
		reflect.Float32, reflect.Float64, reflect.String:
		// Assume scalars are copied by value
	default:
		t.Errorf("%s: unexpected kind: %v", path, a.Kind())
	}
}
