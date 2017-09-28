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
	"encoding/json"
	"math/rand"
	"net/url"
	"reflect"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/api/testing/fuzzer"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	kapitesting "k8s.io/kubernetes/pkg/api/testing"
	k8s_api_v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions"
	utilpointer "k8s.io/kubernetes/pkg/util/pointer"

	// enforce that all types are installed
	_ "k8s.io/kubernetes/pkg/api/testapi"
)

func TestPodLogOptions(t *testing.T) {
	sinceSeconds := int64(1)
	sinceTime := metav1.NewTime(time.Date(2000, 1, 1, 12, 34, 56, 0, time.UTC).Local())
	tailLines := int64(2)
	limitBytes := int64(3)

	versionedLogOptions := &v1.PodLogOptions{
		Container:    "mycontainer",
		Follow:       true,
		Previous:     true,
		SinceSeconds: &sinceSeconds,
		SinceTime:    &sinceTime,
		Timestamps:   true,
		TailLines:    &tailLines,
		LimitBytes:   &limitBytes,
	}
	unversionedLogOptions := &api.PodLogOptions{
		Container:    "mycontainer",
		Follow:       true,
		Previous:     true,
		SinceSeconds: &sinceSeconds,
		SinceTime:    &sinceTime,
		Timestamps:   true,
		TailLines:    &tailLines,
		LimitBytes:   &limitBytes,
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
		convertedLogOptions := &api.PodLogOptions{}
		err := codec.DecodeParameters(expectedParameters, v1.SchemeGroupVersion, convertedLogOptions)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(convertedLogOptions, unversionedLogOptions) {
			t.Fatalf("Unexpected deserialization:\n%s", diff.ObjectGoPrintSideBySide(unversionedLogOptions, convertedLogOptions))
		}
	}
}

// TestPodSpecConversion tests that v1.ServiceAccount is an alias for
// ServiceAccountName.
func TestPodSpecConversion(t *testing.T) {
	name, other := "foo", "bar"

	// Test internal -> v1. Should have both alias (DeprecatedServiceAccount)
	// and new field (ServiceAccountName).
	i := &api.PodSpec{
		ServiceAccountName: name,
	}
	v := v1.PodSpec{}
	if err := legacyscheme.Scheme.Convert(i, &v, nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v.ServiceAccountName != name {
		t.Fatalf("want v1.ServiceAccountName %q, got %q", name, v.ServiceAccountName)
	}
	if v.DeprecatedServiceAccount != name {
		t.Fatalf("want v1.DeprecatedServiceAccount %q, got %q", name, v.DeprecatedServiceAccount)
	}

	// Test v1 -> internal. Either DeprecatedServiceAccount, ServiceAccountName,
	// or both should translate to ServiceAccountName. ServiceAccountName wins
	// if both are set.
	testCases := []*v1.PodSpec{
		// New
		{ServiceAccountName: name},
		// Alias
		{DeprecatedServiceAccount: name},
		// Both: same
		{ServiceAccountName: name, DeprecatedServiceAccount: name},
		// Both: different
		{ServiceAccountName: name, DeprecatedServiceAccount: other},
	}
	for k, v := range testCases {
		got := api.PodSpec{}
		err := legacyscheme.Scheme.Convert(v, &got, nil)
		if err != nil {
			t.Fatalf("unexpected error for case %d: %v", k, err)
		}
		if got.ServiceAccountName != name {
			t.Fatalf("want api.ServiceAccountName %q, got %q", name, got.ServiceAccountName)
		}
	}
}

func TestResourceListConversion(t *testing.T) {
	bigMilliQuantity := resource.NewQuantity(resource.MaxMilliValue, resource.DecimalSI)
	bigMilliQuantity.Add(resource.MustParse("12345m"))

	tests := []struct {
		input    v1.ResourceList
		expected api.ResourceList
	}{
		{ // No changes necessary.
			input: v1.ResourceList{
				v1.ResourceMemory:  resource.MustParse("30M"),
				v1.ResourceCPU:     resource.MustParse("100m"),
				v1.ResourceStorage: resource.MustParse("1G"),
			},
			expected: api.ResourceList{
				api.ResourceMemory:  resource.MustParse("30M"),
				api.ResourceCPU:     resource.MustParse("100m"),
				api.ResourceStorage: resource.MustParse("1G"),
			},
		},
		{ // Nano-scale values should be rounded up to milli-scale.
			input: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("3.000023m"),
				v1.ResourceMemory: resource.MustParse("500.000050m"),
			},
			expected: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("4m"),
				api.ResourceMemory: resource.MustParse("501m"),
			},
		},
		{ // Large values should still be accurate.
			input: v1.ResourceList{
				v1.ResourceCPU:     *bigMilliQuantity.Copy(),
				v1.ResourceStorage: *bigMilliQuantity.Copy(),
			},
			expected: api.ResourceList{
				api.ResourceCPU:     *bigMilliQuantity.Copy(),
				api.ResourceStorage: *bigMilliQuantity.Copy(),
			},
		},
	}

	for i, test := range tests {
		output := api.ResourceList{}

		// defaulting is a separate step from conversion that is applied when reading from the API or from etcd.
		// perform that step explicitly.
		k8s_api_v1.SetDefaults_ResourceList(&test.input)

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
				Replicas:        utilpointer.Int32Ptr(1),
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
	apiObjectFuzzer := fuzzer.FuzzerFor(kapitesting.FuzzerFuncs, rand.NewSource(152), legacyscheme.Codecs)
	for i := 0; i < 100; i++ {
		rc := &v1.ReplicationController{}
		apiObjectFuzzer.Fuzz(rc)
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
		rs := &extensions.ReplicaSet{}
		// Use in.DeepCopy() to avoid sharing pointers with `in`.
		if err := k8s_api_v1.Convert_v1_ReplicationController_to_extensions_ReplicaSet(in.DeepCopy(), rs, nil); err != nil {
			t.Errorf("can't convert RC to RS: %v", err)
			continue
		}
		// Round-trip RS before converting back to RC.
		rs = roundTripRS(t, rs)
		out := &v1.ReplicationController{}
		if err := k8s_api_v1.Convert_extensions_ReplicaSet_to_v1_ReplicationController(rs, out, nil); err != nil {
			t.Errorf("can't convert RS to RC: %v", err)
			continue
		}
		if !apiequality.Semantic.DeepEqual(in, out) {
			instr, _ := json.MarshalIndent(in, "", "  ")
			outstr, _ := json.MarshalIndent(out, "", "  ")
			t.Errorf("RC-RS conversion round-trip failed:\nin:\n%s\nout:\n%s", instr, outstr)
		}
	}
}

func roundTripRS(t *testing.T, rs *extensions.ReplicaSet) *extensions.ReplicaSet {
	codec := legacyscheme.Codecs.LegacyCodec(extensionsv1beta1.SchemeGroupVersion)
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
	obj3 := &extensions.ReplicaSet{}
	err = legacyscheme.Scheme.Convert(obj2, obj3, nil)
	if err != nil {
		t.Errorf("%v\nSource: %#v", err, obj2)
		return nil
	}
	return obj3
}
