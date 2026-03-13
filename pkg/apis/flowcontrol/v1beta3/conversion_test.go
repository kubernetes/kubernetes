/*
Copyright 2023 The Kubernetes Authors.

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

package v1beta3

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/api/flowcontrol/v1beta3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/flowcontrol"
)

func TestConvert_v1beta3_PriorityLevelConfiguration_To_flowcontrol_PriorityLevelConfiguration(t *testing.T) {
	inObjFn := func(shares int32, annotations map[string]string) *v1beta3.PriorityLevelConfiguration {
		return &v1beta3.PriorityLevelConfiguration{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: annotations,
			},
			Spec: v1beta3.PriorityLevelConfigurationSpec{
				Type: v1beta3.PriorityLevelEnablementLimited,
				Limited: &v1beta3.LimitedPriorityLevelConfiguration{
					NominalConcurrencyShares: shares,
					LimitResponse: v1beta3.LimitResponse{
						Type: v1beta3.LimitResponseTypeReject,
					},
				},
			},
		}
	}

	outObjFn := func(shares int32, annotations map[string]string) *flowcontrol.PriorityLevelConfiguration {
		return &flowcontrol.PriorityLevelConfiguration{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: annotations,
			},
			Spec: flowcontrol.PriorityLevelConfigurationSpec{
				Type: flowcontrol.PriorityLevelEnablementLimited,
				Limited: &flowcontrol.LimitedPriorityLevelConfiguration{
					NominalConcurrencyShares: shares,
					LimitResponse: flowcontrol.LimitResponse{
						Type: flowcontrol.LimitResponseTypeReject,
					},
				},
			},
		}
	}

	tests := []struct {
		name     string
		in       *v1beta3.PriorityLevelConfiguration
		expected *flowcontrol.PriorityLevelConfiguration
	}{
		{
			name: "v1beta3 object, the roundtrip annotation is set, NominalConcurrencyShares is zero; the internal object should not have the roundtrip annotation set",
			in: inObjFn(0, map[string]string{
				"foo": "bar",
				v1beta3.PriorityLevelPreserveZeroConcurrencySharesKey: "",
			}),
			expected: outObjFn(0, map[string]string{
				"foo": "bar",
			}),
		},
		{
			name: "v1beta3 object; the internal object should not have the roundtrip annotation set",
			in: &v1beta3.PriorityLevelConfiguration{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"foo": "bar",
						v1beta3.PriorityLevelPreserveZeroConcurrencySharesKey: "",
					},
				},
			},
			expected: &flowcontrol.PriorityLevelConfiguration{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"foo": "bar",
					},
				},
			},
		},
		{
			name: "v1beta3 object, the roundtrip annotation is not set, NominalConcurrencyShares is zero; the internal object should not have the roundtrip annotation set",
			in: inObjFn(0, map[string]string{
				"foo": "bar",
			}),
			expected: outObjFn(0, map[string]string{
				"foo": "bar",
			}),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			copy := test.in.DeepCopy()

			out := &flowcontrol.PriorityLevelConfiguration{}
			if err := Convert_v1beta3_PriorityLevelConfiguration_To_flowcontrol_PriorityLevelConfiguration(test.in, out, nil); err != nil {
				t.Errorf("Expected no error, but got: %v", err)
			}
			if !cmp.Equal(test.expected, out) {
				t.Errorf("Expected a match, diff: %s", cmp.Diff(test.expected, out))
			}
			if want, got := copy.ObjectMeta.Annotations, test.in.ObjectMeta.Annotations; !cmp.Equal(want, got) {
				t.Errorf("Did not expect the 'Annotations' field of the source to be mutated, diff: %s", cmp.Diff(want, got))
			}
		})
	}
}

func TestConvert_flowcontrol_PriorityLevelConfiguration_To_v1beta3_PriorityLevelConfiguration(t *testing.T) {
	inObjFn := func(shares int32, annotations map[string]string) *flowcontrol.PriorityLevelConfiguration {
		return &flowcontrol.PriorityLevelConfiguration{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: annotations,
			},
			Spec: flowcontrol.PriorityLevelConfigurationSpec{
				Type: flowcontrol.PriorityLevelEnablementLimited,
				Limited: &flowcontrol.LimitedPriorityLevelConfiguration{
					NominalConcurrencyShares: shares,
					LimitResponse: flowcontrol.LimitResponse{
						Type: flowcontrol.LimitResponseTypeReject,
					},
				},
			},
		}
	}

	outObjFn := func(shares int32, annotations map[string]string) *v1beta3.PriorityLevelConfiguration {
		return &v1beta3.PriorityLevelConfiguration{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: annotations,
			},
			Spec: v1beta3.PriorityLevelConfigurationSpec{
				Type: v1beta3.PriorityLevelEnablementLimited,
				Limited: &v1beta3.LimitedPriorityLevelConfiguration{
					NominalConcurrencyShares: shares,
					LimitResponse: v1beta3.LimitResponse{
						Type: v1beta3.LimitResponseTypeReject,
					},
				},
			},
		}
	}

	tests := []struct {
		name     string
		in       *flowcontrol.PriorityLevelConfiguration
		expected *v1beta3.PriorityLevelConfiguration
	}{
		{
			name: "internal object, NominalConcurrencyShares is 0; v1beta3 object should have the roundtrip annotation set",
			in: inObjFn(0, map[string]string{
				"foo": "bar",
			}),
			expected: outObjFn(0, map[string]string{
				"foo": "bar",
				v1beta3.PriorityLevelPreserveZeroConcurrencySharesKey: "",
			}),
		},
		{
			name: "internal object, NominalConcurrencyShares is not 0; v1beta3 object should not have the roundtrip annotation set",
			in: inObjFn(1, map[string]string{
				"foo": "bar",
			}),
			expected: outObjFn(1, map[string]string{
				"foo": "bar",
			}),
		},
		{
			name: "internal object, the roundtrip annotation is set, NominalConcurrencyShares is 0, no error expected",
			in: inObjFn(0, map[string]string{
				"foo": "bar",
				v1beta3.PriorityLevelPreserveZeroConcurrencySharesKey: "",
			}),
			expected: outObjFn(0, map[string]string{
				"foo": "bar",
				v1beta3.PriorityLevelPreserveZeroConcurrencySharesKey: "",
			}),
		},
		{
			name: "internal object, the roundtrip annotation is set with a non-empty value, NominalConcurrencyShares is 0, the annotation value should be preserved",
			in: inObjFn(0, map[string]string{
				"foo": "bar",
				v1beta3.PriorityLevelPreserveZeroConcurrencySharesKey: "non-empty",
			}),
			expected: outObjFn(0, map[string]string{
				"foo": "bar",
				v1beta3.PriorityLevelPreserveZeroConcurrencySharesKey: "non-empty",
			}),
		},
		{
			name: "internal object, the roundtrip annotation is set with a non-empty value, NominalConcurrencyShares is not 0, the annotation value should be preserved",
			in: inObjFn(1, map[string]string{
				"foo": "bar",
				v1beta3.PriorityLevelPreserveZeroConcurrencySharesKey: "non-empty",
			}),
			expected: outObjFn(1, map[string]string{
				"foo": "bar",
				v1beta3.PriorityLevelPreserveZeroConcurrencySharesKey: "non-empty",
			}),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			copy := test.in.DeepCopy()

			out := &v1beta3.PriorityLevelConfiguration{}
			if err := Convert_flowcontrol_PriorityLevelConfiguration_To_v1beta3_PriorityLevelConfiguration(test.in, out, nil); err != nil {
				t.Errorf("Expected no error, but got: %v", err)
			}

			if !cmp.Equal(test.expected, out) {
				t.Errorf("Expected a match, diff: %s", cmp.Diff(test.expected, out))
			}
			if want, got := copy.ObjectMeta.Annotations, test.in.ObjectMeta.Annotations; !cmp.Equal(want, got) {
				t.Errorf("Did not expect the 'Annotations' field of the source to be mutated, diff: %s", cmp.Diff(want, got))
			}
		})
	}
}
