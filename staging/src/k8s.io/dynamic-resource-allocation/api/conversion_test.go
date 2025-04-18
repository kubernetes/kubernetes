/*
Copyright 2025 The Kubernetes Authors.

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

package api_test

import (
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"k8s.io/api/resource/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	draapi "k8s.io/dynamic-resource-allocation/api"
	"k8s.io/utils/ptr"
)

func TestConversion(t *testing.T) {
	testcases := map[string]struct {
		in        v1beta1.ResourceSlice
		expectOut draapi.ResourceSlice
		expectErr string
	}{
		"minimal": {
			in: toV1beta1ResourceSlice(
				[]v1beta1.CounterSet{},
				&v1beta1.ResourceSliceMixins{},
				[]v1beta1.Device{},
			),
			expectOut: toResourceSlice(
				[]draapi.CounterSet{},
				[]draapi.Device{},
			),
		},
		"preserve-nil-lists": {
			in: toV1beta1ResourceSlice(
				[]v1beta1.CounterSet{
					{
						Name:     "counter-set-1",
						Counters: nil,
					},
				},
				nil,
				[]v1beta1.Device{
					{
						Name: "device-1",
						Basic: &v1beta1.BasicDevice{
							Attributes: nil,
							Capacity:   nil,
						},
					},
				},
			),
			expectOut: toResourceSlice(
				[]draapi.CounterSet{
					{
						Name:     draapi.MakeUniqueString("counter-set-1"),
						Counters: nil,
					},
				},
				[]draapi.Device{
					{
						Name: draapi.MakeUniqueString("device-1"),
						Basic: &draapi.BasicDevice{
							Attributes: nil,
							Capacity:   nil,
						},
					},
				},
			),
		},
		"counter-set-mixins-are-added": {
			in: toV1beta1ResourceSlice(
				[]v1beta1.CounterSet{
					{
						Name: "counter-set-1",
						Counters: map[string]v1beta1.Counter{
							"mem-1": {
								Value: resource.MustParse("10Gi"),
							},
						},
						Includes: []v1beta1.CounterSetMixinRef{
							{
								Name: "counter-set-mixin-1",
							},
						},
					},
				},
				&v1beta1.ResourceSliceMixins{
					CounterSet: []v1beta1.CounterSetMixin{
						{
							Name: "counter-set-mixin-1",
							Counters: map[string]v1beta1.Counter{
								"mem-2": {
									Value: resource.MustParse("15Gi"),
								},
							},
						},
					},
				},
				[]v1beta1.Device{},
			),
			expectOut: toResourceSlice(
				[]draapi.CounterSet{
					{
						Name: draapi.MakeUniqueString("counter-set-1"),
						Counters: map[string]draapi.Counter{
							"mem-1": {
								Value: resource.MustParse("10Gi"),
							},
							"mem-2": {
								Value: resource.MustParse("15Gi"),
							},
						},
					},
				},
				[]draapi.Device{},
			),
		},
		"counter-set-source-overrides-mixins": {
			in: toV1beta1ResourceSlice(
				[]v1beta1.CounterSet{
					{
						Name: "counter-set-1",
						Counters: map[string]v1beta1.Counter{
							"mem-1": {
								Value: resource.MustParse("10Gi"),
							},
						},
						Includes: []v1beta1.CounterSetMixinRef{
							{
								Name: "counter-set-mixin-1",
							},
						},
					},
				},
				&v1beta1.ResourceSliceMixins{
					CounterSet: []v1beta1.CounterSetMixin{
						{
							Name: "counter-set-mixin-1",
							Counters: map[string]v1beta1.Counter{
								"mem-1": {
									Value: resource.MustParse("15Gi"),
								},
							},
						},
					},
				},
				[]v1beta1.Device{},
			),
			expectOut: toResourceSlice(
				[]draapi.CounterSet{
					{
						Name: draapi.MakeUniqueString("counter-set-1"),
						Counters: map[string]draapi.Counter{
							"mem-1": {
								Value: resource.MustParse("10Gi"),
							},
						},
					},
				},
				[]draapi.Device{},
			),
		},
		"counter-set-mixin-overrides-mixins": {
			in: toV1beta1ResourceSlice(
				[]v1beta1.CounterSet{
					{
						Name:     "counter-set-1",
						Counters: nil,
						Includes: []v1beta1.CounterSetMixinRef{
							{
								Name: "counter-set-mixin-1",
							},
							{
								Name: "counter-set-mixin-2",
							},
						},
					},
				},
				&v1beta1.ResourceSliceMixins{
					CounterSet: []v1beta1.CounterSetMixin{
						{
							Name: "counter-set-mixin-1",
							Counters: map[string]v1beta1.Counter{
								"mem-1": {
									Value: resource.MustParse("15Gi"),
								},
							},
						},
						{
							Name: "counter-set-mixin-2",
							Counters: map[string]v1beta1.Counter{
								"mem-1": {
									Value: resource.MustParse("20Gi"),
								},
							},
						},
					},
				},
				[]v1beta1.Device{},
			),
			expectOut: toResourceSlice(
				[]draapi.CounterSet{
					{
						Name: draapi.MakeUniqueString("counter-set-1"),
						Counters: map[string]draapi.Counter{
							"mem-1": {
								Value: resource.MustParse("20Gi"),
							},
						},
					},
				},
				[]draapi.Device{},
			),
		},
		"device-mixins-are-added": {
			in: toV1beta1ResourceSlice(
				[]v1beta1.CounterSet{},
				&v1beta1.ResourceSliceMixins{
					Device: []v1beta1.DeviceMixin{
						{
							Name: "device-mixin-1",
							Attributes: map[v1beta1.QualifiedName]v1beta1.DeviceAttribute{
								"attr-1": {
									StringValue: ptr.To("value-1"),
								},
							},
							Capacity: map[v1beta1.QualifiedName]v1beta1.DeviceCapacity{
								"mem": {
									Value: resource.MustParse("10Gi"),
								},
							},
						},
					},
				},
				[]v1beta1.Device{
					{
						Name: "device-1",
						Basic: &v1beta1.BasicDevice{
							Attributes: map[v1beta1.QualifiedName]v1beta1.DeviceAttribute{
								"attr-2": {
									StringValue: ptr.To("value-2"),
								},
							},
							Capacity: map[v1beta1.QualifiedName]v1beta1.DeviceCapacity{
								"other-mem": {
									Value: resource.MustParse("20Gi"),
								},
							},
							Includes: []v1beta1.DeviceMixinRef{
								{
									Name: "device-mixin-1",
								},
							},
						},
					},
				},
			),
			expectOut: toResourceSlice(
				[]draapi.CounterSet{},
				[]draapi.Device{
					{
						Name: draapi.MakeUniqueString("device-1"),
						Basic: &draapi.BasicDevice{
							Attributes: map[draapi.QualifiedName]draapi.DeviceAttribute{
								"attr-1": {
									StringValue: ptr.To("value-1"),
								},
								"attr-2": {
									StringValue: ptr.To("value-2"),
								},
							},
							Capacity: map[draapi.QualifiedName]draapi.DeviceCapacity{
								"mem": {
									Value: resource.MustParse("10Gi"),
								},
								"other-mem": {
									Value: resource.MustParse("20Gi"),
								},
							},
						},
					},
				},
			),
		},
		"device-source-overrides-mixins": {
			in: toV1beta1ResourceSlice(
				[]v1beta1.CounterSet{},
				&v1beta1.ResourceSliceMixins{
					Device: []v1beta1.DeviceMixin{
						{
							Name: "device-mixin-1",
							Attributes: map[v1beta1.QualifiedName]v1beta1.DeviceAttribute{
								"attr-1": {
									StringValue: ptr.To("value-1"),
								},
							},
							Capacity: map[v1beta1.QualifiedName]v1beta1.DeviceCapacity{
								"mem": {
									Value: resource.MustParse("10Gi"),
								},
							},
						},
					},
				},
				[]v1beta1.Device{
					{
						Name: "device-1",
						Basic: &v1beta1.BasicDevice{
							Attributes: map[v1beta1.QualifiedName]v1beta1.DeviceAttribute{
								"attr-1": {
									StringValue: ptr.To("other-value"),
								},
							},
							Capacity: map[v1beta1.QualifiedName]v1beta1.DeviceCapacity{
								"mem": {
									Value: resource.MustParse("20Gi"),
								},
							},
							Includes: []v1beta1.DeviceMixinRef{
								{
									Name: "device-mixin-1",
								},
							},
						},
					},
				},
			),
			expectOut: toResourceSlice(
				[]draapi.CounterSet{},
				[]draapi.Device{
					{
						Name: draapi.MakeUniqueString("device-1"),
						Basic: &draapi.BasicDevice{
							Attributes: map[draapi.QualifiedName]draapi.DeviceAttribute{
								"attr-1": {
									StringValue: ptr.To("other-value"),
								},
							},
							Capacity: map[draapi.QualifiedName]draapi.DeviceCapacity{
								"mem": {
									Value: resource.MustParse("20Gi"),
								},
							},
						},
					},
				},
			),
		},
		"device-mixin-overrides-mixins": {
			in: toV1beta1ResourceSlice(
				[]v1beta1.CounterSet{},
				&v1beta1.ResourceSliceMixins{
					Device: []v1beta1.DeviceMixin{
						{
							Name: "device-mixin-1",
							Attributes: map[v1beta1.QualifiedName]v1beta1.DeviceAttribute{
								"attr-1": {
									StringValue: ptr.To("value-1"),
								},
							},
							Capacity: map[v1beta1.QualifiedName]v1beta1.DeviceCapacity{
								"mem": {
									Value: resource.MustParse("10Gi"),
								},
							},
						},
						{
							Name: "device-mixin-2",
							Attributes: map[v1beta1.QualifiedName]v1beta1.DeviceAttribute{
								"attr-1": {
									StringValue: ptr.To("other-value"),
								},
							},
							Capacity: map[v1beta1.QualifiedName]v1beta1.DeviceCapacity{
								"mem": {
									Value: resource.MustParse("20Gi"),
								},
							},
						},
					},
				},
				[]v1beta1.Device{
					{
						Name: "device-1",
						Basic: &v1beta1.BasicDevice{
							Includes: []v1beta1.DeviceMixinRef{
								{
									Name: "device-mixin-1",
								},
								{
									Name: "device-mixin-2",
								},
							},
						},
					},
				},
			),
			expectOut: toResourceSlice(
				[]draapi.CounterSet{},
				[]draapi.Device{
					{
						Name: draapi.MakeUniqueString("device-1"),
						Basic: &draapi.BasicDevice{
							Attributes: map[draapi.QualifiedName]draapi.DeviceAttribute{
								"attr-1": {
									StringValue: ptr.To("other-value"),
								},
							},
							Capacity: map[draapi.QualifiedName]draapi.DeviceCapacity{
								"mem": {
									Value: resource.MustParse("20Gi"),
								},
							},
						},
					},
				},
			),
		},
		"consumes-counters-mixins-are-added": {
			in: toV1beta1ResourceSlice(
				[]v1beta1.CounterSet{},
				&v1beta1.ResourceSliceMixins{
					DeviceCounterConsumption: []v1beta1.DeviceCounterConsumptionMixin{
						{
							Name: "device-counter-consumption-mixin-1",
							Counters: map[string]v1beta1.Counter{
								"mem": {
									Value: resource.MustParse("10Gi"),
								},
							},
						},
					},
				},
				[]v1beta1.Device{
					{
						Name: "device-1",
						Basic: &v1beta1.BasicDevice{
							ConsumesCounters: []v1beta1.DeviceCounterConsumption{
								{
									CounterSet: "counter-set-1",
									Counters: map[string]v1beta1.Counter{
										"other-mem": {
											Value: resource.MustParse("20Gi"),
										},
									},
									Includes: []v1beta1.DeviceCounterConsumptionMixinRef{
										{
											Name: "device-counter-consumption-mixin-1",
										},
									},
								},
							},
						},
					},
				},
			),
			expectOut: toResourceSlice(
				[]draapi.CounterSet{},
				[]draapi.Device{
					{
						Name: draapi.MakeUniqueString("device-1"),
						Basic: &draapi.BasicDevice{
							ConsumesCounters: []draapi.DeviceCounterConsumption{
								{
									CounterSet: draapi.MakeUniqueString("counter-set-1"),
									Counters: map[string]draapi.Counter{
										"mem": {
											Value: resource.MustParse("10Gi"),
										},
										"other-mem": {
											Value: resource.MustParse("20Gi"),
										},
									},
								},
							},
						},
					},
				},
			),
		},
		"consumes-counters-source-overrides-mixins": {
			in: toV1beta1ResourceSlice(
				[]v1beta1.CounterSet{},
				&v1beta1.ResourceSliceMixins{
					DeviceCounterConsumption: []v1beta1.DeviceCounterConsumptionMixin{
						{
							Name: "device-counter-consumption-mixin-1",
							Counters: map[string]v1beta1.Counter{
								"mem": {
									Value: resource.MustParse("10Gi"),
								},
							},
						},
					},
				},
				[]v1beta1.Device{
					{
						Name: "device-1",
						Basic: &v1beta1.BasicDevice{
							ConsumesCounters: []v1beta1.DeviceCounterConsumption{
								{
									CounterSet: "counter-set-1",
									Counters: map[string]v1beta1.Counter{
										"mem": {
											Value: resource.MustParse("20Gi"),
										},
									},
									Includes: []v1beta1.DeviceCounterConsumptionMixinRef{
										{
											Name: "device-counter-consumption-mixin-1",
										},
									},
								},
							},
						},
					},
				},
			),
			expectOut: toResourceSlice(
				[]draapi.CounterSet{},
				[]draapi.Device{
					{
						Name: draapi.MakeUniqueString("device-1"),
						Basic: &draapi.BasicDevice{
							ConsumesCounters: []draapi.DeviceCounterConsumption{
								{
									CounterSet: draapi.MakeUniqueString("counter-set-1"),
									Counters: map[string]draapi.Counter{
										"mem": {
											Value: resource.MustParse("20Gi"),
										},
									},
								},
							},
						},
					},
				},
			),
		},
		"consumes-counters-mixin-overrides-mixins": {
			in: toV1beta1ResourceSlice(
				[]v1beta1.CounterSet{},
				&v1beta1.ResourceSliceMixins{
					DeviceCounterConsumption: []v1beta1.DeviceCounterConsumptionMixin{
						{
							Name: "device-counter-consumption-mixin-1",
							Counters: map[string]v1beta1.Counter{
								"mem": {
									Value: resource.MustParse("10Gi"),
								},
							},
						},
						{
							Name: "device-counter-consumption-mixin-2",
							Counters: map[string]v1beta1.Counter{
								"mem": {
									Value: resource.MustParse("20Gi"),
								},
							},
						},
					},
				},
				[]v1beta1.Device{
					{
						Name: "device-1",
						Basic: &v1beta1.BasicDevice{
							ConsumesCounters: []v1beta1.DeviceCounterConsumption{
								{
									CounterSet: "counter-set-1",
									Includes: []v1beta1.DeviceCounterConsumptionMixinRef{
										{
											Name: "device-counter-consumption-mixin-1",
										},
										{
											Name: "device-counter-consumption-mixin-2",
										},
									},
								},
							},
						},
					},
				},
			),
			expectOut: toResourceSlice(
				[]draapi.CounterSet{},
				[]draapi.Device{
					{
						Name: draapi.MakeUniqueString("device-1"),
						Basic: &draapi.BasicDevice{
							ConsumesCounters: []draapi.DeviceCounterConsumption{
								{
									CounterSet: draapi.MakeUniqueString("counter-set-1"),
									Counters: map[string]draapi.Counter{
										"mem": {
											Value: resource.MustParse("20Gi"),
										},
									},
								},
							},
						},
					},
				},
			),
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			var out draapi.ResourceSlice
			scope := draapi.SliceScope{
				SliceContext: draapi.SliceContext{
					Slice: &tc.in,
				},
			}
			err := draapi.Convert_v1beta1_ResourceSlice_To_api_ResourceSlice(&tc.in, &out, scope)
			if err != nil {
				if len(tc.expectErr) == 0 {
					t.Fatalf("unexpected error %v", err)
				}
				if !strings.Contains(err.Error(), tc.expectErr) {
					t.Fatalf("expected error %s, got %v", tc.expectErr, err)
				}
				return
			}
			if len(tc.expectErr) > 0 {
				t.Fatalf("expected error %s, got none", tc.expectErr)
			}
			if !reflect.DeepEqual(out, tc.expectOut) {
				t.Fatalf("unexpected result:\n %s", cmp.Diff(tc.expectOut, out, cmpopts.EquateComparable(draapi.UniqueString{})))
			}
		})
	}
}

func toV1beta1ResourceSlice(sharedCounters []v1beta1.CounterSet, mixins *v1beta1.ResourceSliceMixins, devices []v1beta1.Device) v1beta1.ResourceSlice {
	return v1beta1.ResourceSlice{
		Spec: v1beta1.ResourceSliceSpec{
			Driver: "driver-a",
			Pool: v1beta1.ResourcePool{
				Name:               "pool-1",
				Generation:         1,
				ResourceSliceCount: 1,
			},
			NodeName:       "node-1",
			SharedCounters: sharedCounters,
			Mixins:         mixins,
			Devices:        devices,
		},
	}
}

func toResourceSlice(sharedCounters []draapi.CounterSet, devices []draapi.Device) draapi.ResourceSlice {
	return draapi.ResourceSlice{
		Spec: draapi.ResourceSliceSpec{
			Driver: draapi.MakeUniqueString("driver-a"),
			Pool: draapi.ResourcePool{
				Name:               draapi.MakeUniqueString("pool-1"),
				Generation:         1,
				ResourceSliceCount: 1,
			},
			NodeName:       draapi.MakeUniqueString("node-1"),
			SharedCounters: sharedCounters,
			Devices:        devices,
		},
	}
}
