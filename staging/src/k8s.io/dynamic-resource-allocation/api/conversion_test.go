package api_test

import (
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	resourcev1beta1 "k8s.io/api/resource/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	draapi "k8s.io/dynamic-resource-allocation/api"
)

func TestConversion(t *testing.T) {
	testcases := map[string]struct {
		in        resourcev1beta1.ResourceSlice
		expectOut draapi.ResourceSlice
		expectErr string
	}{
		"simple-resourceslice-with-basic-devices": {
			in: resourcev1beta1.ResourceSlice{
				Spec: resourcev1beta1.ResourceSliceSpec{
					Driver: "driver-a",
					Pool: resourcev1beta1.ResourcePool{
						Name:               "pool-1",
						Generation:         1,
						ResourceSliceCount: 1,
					},
					NodeName: "node-1",
					Devices: []resourcev1beta1.Device{
						{
							Name: "device-1",
							Basic: &resourcev1beta1.BasicDevice{
								Attributes: map[resourcev1beta1.QualifiedName]resourcev1beta1.DeviceAttribute{
									resourcev1beta1.QualifiedName("foo"): {
										IntValue: func() *int64 {
											i := int64(42)
											return &i
										}(),
									},
								},
								Capacity: map[resourcev1beta1.QualifiedName]resourcev1beta1.DeviceCapacity{
									resourcev1beta1.QualifiedName("memory"): {
										Value: resource.MustParse("50Gi"),
									},
								},
							},
						},
						{
							Name:  "device-2",
							Basic: &resourcev1beta1.BasicDevice{},
						},
					},
				},
			},
			expectOut: draapi.ResourceSlice{
				Spec: draapi.ResourceSliceSpec{
					Driver: draapi.MakeUniqueString("driver-a"),
					Pool: draapi.ResourcePool{
						Name:               draapi.MakeUniqueString("pool-1"),
						Generation:         1,
						ResourceSliceCount: 1,
					},
					NodeName: draapi.MakeUniqueString("node-1"),
					Devices: []draapi.Device{
						{
							Name: draapi.MakeUniqueString("device-1"),
							Basic: &draapi.BasicDevice{
								Attributes: map[draapi.QualifiedName]draapi.DeviceAttribute{
									draapi.QualifiedName("foo"): {
										IntValue: func() *int64 {
											i := int64(42)
											return &i
										}(),
									},
								},
								Capacity: map[draapi.QualifiedName]draapi.DeviceCapacity{
									draapi.QualifiedName("memory"): {
										Value: resource.MustParse("50Gi"),
									},
								},
							},
						},
						{
							Name:  draapi.MakeUniqueString("device-2"),
							Basic: &draapi.BasicDevice{},
						},
					},
				},
			},
		},

		"resourceslice-with-composite-devices-and-device-mixin": {
			in: resourcev1beta1.ResourceSlice{
				Spec: resourcev1beta1.ResourceSliceSpec{
					Driver: "driver-a",
					Pool: resourcev1beta1.ResourcePool{
						Name:               "pool-1",
						Generation:         2,
						ResourceSliceCount: 3,
					},
					NodeName: "node-1",
					Mixins: &resourcev1beta1.ResourceSliceMixins{
						Device: []resourcev1beta1.DeviceMixin{
							{
								Name: "device-mixin-1",
								Composite: &resourcev1beta1.CompositeDeviceMixin{
									Attributes: map[resourcev1beta1.QualifiedName]resourcev1beta1.DeviceAttribute{
										resourcev1beta1.QualifiedName("attribute-1"): {
											IntValue: func() *int64 {
												val := int64(42)
												return &val
											}(),
										},
										resourcev1beta1.QualifiedName("attribute-2"): {
											StringValue: func() *string {
												val := "foo"
												return &val
											}(),
										},
									},
									Capacity: map[resourcev1beta1.QualifiedName]resourcev1beta1.DeviceCapacity{
										resourcev1beta1.QualifiedName("memory"): {
											Value: resource.MustParse("50Gi"),
										},
										resourcev1beta1.QualifiedName("processors"): {
											Value: resource.MustParse("42"),
										},
									},
								},
							},
						},
					},
					Devices: []resourcev1beta1.Device{
						{
							Name: "device-1",
							Composite: &resourcev1beta1.CompositeDevice{
								Includes: []resourcev1beta1.DeviceMixinRef{
									{
										Name: "device-mixin-1",
									},
								},
								Attributes: map[resourcev1beta1.QualifiedName]resourcev1beta1.DeviceAttribute{
									resourcev1beta1.QualifiedName("attribute-1"): {
										IntValue: func() *int64 {
											i := int64(24)
											return &i
										}(),
									},
									resourcev1beta1.QualifiedName("attribute-3"): {
										BoolValue: func() *bool {
											val := true
											return &val
										}(),
									},
								},
								Capacity: map[resourcev1beta1.QualifiedName]resourcev1beta1.DeviceCapacity{
									resourcev1beta1.QualifiedName("memory"): {
										Value: resource.MustParse("20Gi"),
									},
									resourcev1beta1.QualifiedName("gpus"): {
										Value: resource.MustParse("24"),
									},
								},
							},
						},
					},
				},
			},
			expectOut: draapi.ResourceSlice{
				Spec: draapi.ResourceSliceSpec{
					Driver: draapi.MakeUniqueString("driver-a"),
					Pool: draapi.ResourcePool{
						Name:               draapi.MakeUniqueString("pool-1"),
						Generation:         2,
						ResourceSliceCount: 3,
					},
					NodeName: draapi.MakeUniqueString("node-1"),
					Devices: []draapi.Device{
						{
							Name: draapi.MakeUniqueString("device-1"),
							Composite: &draapi.CompositeDevice{
								Includes: []draapi.DeviceMixinRef{
									{
										Name: draapi.MakeUniqueString("device-mixin-1"),
									},
								},
								Attributes: map[draapi.QualifiedName]draapi.DeviceAttribute{
									draapi.QualifiedName("attribute-1"): {
										IntValue: func() *int64 {
											i := int64(24)
											return &i
										}(),
									},
									draapi.QualifiedName("attribute-2"): {
										StringValue: func() *string {
											val := "foo"
											return &val
										}(),
									},
									draapi.QualifiedName("attribute-3"): {
										BoolValue: func() *bool {
											val := true
											return &val
										}(),
									},
								},
								Capacity: map[draapi.QualifiedName]draapi.DeviceCapacity{
									draapi.QualifiedName("memory"): {
										Value: resource.MustParse("20Gi"),
									},
									draapi.QualifiedName("processors"): {
										Value: resource.MustParse("42"),
									},
									draapi.QualifiedName("gpus"): {
										Value: resource.MustParse("24"),
									},
								},
							},
						},
					},
				},
			},
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			var out draapi.ResourceSlice
			err := draapi.Convert_v1beta1_ResourceSlice_To_api_ResourceSlice(&tc.in, &out, nil)
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
