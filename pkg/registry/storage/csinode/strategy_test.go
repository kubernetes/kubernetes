/*
Copyright 2019 The Kubernetes Authors.

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

package csinode

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/features"
	utilpointer "k8s.io/utils/pointer"
)

func TestPrepareForCreate(t *testing.T) {
	valid := getValidCSINode("foo")
	emptyAllocatable := &storage.CSINode{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: storage.CSINodeSpec{
			Drivers: []storage.CSINodeDriver{
				{
					Name:         "valid-driver-name",
					NodeID:       "valid-node",
					TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
				},
			},
		},
	}

	volumeLimitsEnabledCases := []struct {
		name     string
		obj      *storage.CSINode
		expected *storage.CSINode
	}{
		{
			"empty allocatable",
			emptyAllocatable,
			emptyAllocatable,
		},
		{
			"valid allocatable",
			valid,
			valid,
		},
	}

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AttachVolumeLimit, true)()
	for _, test := range volumeLimitsEnabledCases {
		t.Run(test.name, func(t *testing.T) {
			testPrepareForCreate(t, test.obj, test.expected)
		})
	}

	volumeLimitsDisabledCases := []struct {
		name     string
		obj      *storage.CSINode
		expected *storage.CSINode
	}{
		{
			"empty allocatable",
			emptyAllocatable,
			emptyAllocatable,
		},
		{
			"drop allocatable",
			valid,
			emptyAllocatable,
		},
	}

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AttachVolumeLimit, false)()
	for _, test := range volumeLimitsDisabledCases {
		t.Run(test.name, func(t *testing.T) {
			testPrepareForCreate(t, test.obj, test.expected)
		})
	}
}

func testPrepareForCreate(t *testing.T, obj, expected *storage.CSINode) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &genericapirequest.RequestInfo{
		APIGroup:   "storage.k8s.io",
		APIVersion: "v1beta1",
		Resource:   "csinodes",
	})
	Strategy.PrepareForCreate(ctx, obj)
	if !reflect.DeepEqual(*expected, *obj) {
		t.Errorf("Object mismatch! Expected:\n%#v\ngot:\n%#v", *expected, *obj)
	}
}

func TestPrepareForUpdate(t *testing.T) {
	valid := getValidCSINode("foo")
	differentAllocatable := &storage.CSINode{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: storage.CSINodeSpec{
			Drivers: []storage.CSINodeDriver{
				{
					Name:         "valid-driver-name",
					NodeID:       "valid-node",
					TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					Allocatable:  &storage.VolumeNodeResources{Count: utilpointer.Int32Ptr(20)},
				},
			},
		},
	}
	emptyAllocatable := &storage.CSINode{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: storage.CSINodeSpec{
			Drivers: []storage.CSINodeDriver{
				{
					Name:         "valid-driver-name",
					NodeID:       "valid-node",
					TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
				},
			},
		},
	}

	volumeLimitsEnabledCases := []struct {
		name     string
		old      *storage.CSINode
		new      *storage.CSINode
		expected *storage.CSINode
	}{
		{
			"allow empty allocatable when it's not set",
			emptyAllocatable,
			emptyAllocatable,
			emptyAllocatable,
		},
		{
			"allow valid allocatable when it's already set",
			valid,
			differentAllocatable,
			differentAllocatable,
		},
		{
			"allow valid allocatable when it's not set",
			emptyAllocatable,
			valid,
			valid,
		},
	}

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AttachVolumeLimit, true)()
	for _, test := range volumeLimitsEnabledCases {
		t.Run(test.name, func(t *testing.T) {
			testPrepareForUpdate(t, test.new, test.old, test.expected)
		})
	}

	volumeLimitsDisabledCases := []struct {
		name     string
		old      *storage.CSINode
		new      *storage.CSINode
		expected *storage.CSINode
	}{
		{
			"allow empty allocatable when it's not set",
			emptyAllocatable,
			emptyAllocatable,
			emptyAllocatable,
		},
		{
			"drop allocatable when it's not set",
			emptyAllocatable,
			valid,
			emptyAllocatable,
		},
	}

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AttachVolumeLimit, false)()
	for _, test := range volumeLimitsDisabledCases {
		t.Run(test.name, func(t *testing.T) {
			testPrepareForUpdate(t, test.new, test.old, test.expected)
		})
	}
}

func testPrepareForUpdate(t *testing.T, obj, old, expected *storage.CSINode) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &genericapirequest.RequestInfo{
		APIGroup:   "storage.k8s.io",
		APIVersion: "v1beta1",
		Resource:   "csinodes",
	})
	Strategy.PrepareForUpdate(ctx, obj, old)
	if !reflect.DeepEqual(*expected, *obj) {
		t.Errorf("Object mismatch! Expected:\n%#v\ngot:\n%#v", *expected, *obj)
	}
}

func TestCSINodeStrategy(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &genericapirequest.RequestInfo{
		APIGroup:   "storage.k8s.io",
		APIVersion: "v1beta1",
		Resource:   "csinodes",
	})
	if Strategy.NamespaceScoped() {
		t.Errorf("CSINode must not be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("CSINode should not allow create on update")
	}

	csiNode := getValidCSINode("valid-csinode")

	Strategy.PrepareForCreate(ctx, csiNode)

	errs := Strategy.Validate(ctx, csiNode)
	if len(errs) != 0 {
		t.Errorf("unexpected error validating %v", errs)
	}

	// Update of spec is allowed
	newCSINode := csiNode.DeepCopy()
	newCSINode.Spec.Drivers[0].NodeID = "valid-node-2"

	Strategy.PrepareForUpdate(ctx, newCSINode, csiNode)

	errs = Strategy.ValidateUpdate(ctx, newCSINode, csiNode)
	if len(errs) == 0 {
		t.Errorf("expected validation error")
	}
}

func TestCSINodeValidation(t *testing.T) {
	tests := []struct {
		name        string
		csiNode     *storage.CSINode
		expectError bool
	}{
		{
			"valid csinode",
			getValidCSINode("foo"),
			false,
		},
		{
			"valid csinode with empty allocatable",
			&storage.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "valid-driver-name",
							NodeID:       "valid-node",
							TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
						},
					},
				},
			},
			false,
		},
		{
			"valid csinode with missing volume limits",
			&storage.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "valid-driver-name",
							NodeID:       "valid-node",
							TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
							Allocatable:  &storage.VolumeNodeResources{Count: nil},
						},
					},
				},
			},
			false,
		},
		{
			"invalid driver name",
			&storage.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "$csi-driver@",
							NodeID:       "valid-node",
							TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
							Allocatable:  &storage.VolumeNodeResources{Count: utilpointer.Int32Ptr(10)},
						},
					},
				},
			},
			true,
		},
		{
			"empty node id",
			&storage.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "valid-driver-name",
							NodeID:       "",
							TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
							Allocatable:  &storage.VolumeNodeResources{Count: utilpointer.Int32Ptr(10)},
						},
					},
				},
			},
			true,
		},
		{
			"invalid allocatable with negative volumes limit",
			&storage.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "valid-driver-name",
							NodeID:       "valid-node",
							TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
							Allocatable:  &storage.VolumeNodeResources{Count: utilpointer.Int32Ptr(-1)},
						},
					},
				},
			},
			true,
		},
		{
			"invalid topology keys",
			&storage.CSINode{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.CSINodeSpec{
					Drivers: []storage.CSINodeDriver{
						{
							Name:         "valid-driver-name",
							NodeID:       "valid-node",
							TopologyKeys: []string{"company.com/zone1", ""},
							Allocatable:  &storage.VolumeNodeResources{Count: utilpointer.Int32Ptr(10)},
						},
					},
				},
			},
			true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {

			testValidation := func(csiNode *storage.CSINode, apiVersion string) field.ErrorList {
				ctx := genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &genericapirequest.RequestInfo{
					APIGroup:   "storage.k8s.io",
					APIVersion: "v1beta1",
					Resource:   "csinodes",
				})
				return Strategy.Validate(ctx, csiNode)
			}

			betaErr := testValidation(test.csiNode, "v1beta1")
			if len(betaErr) > 0 && !test.expectError {
				t.Errorf("Validation of v1beta1 object failed: %+v", betaErr)
			}
			if len(betaErr) == 0 && test.expectError {
				t.Errorf("Validation of v1beta1 object unexpectedly succeeded")
			}
		})
	}
}

func getValidCSINode(name string) *storage.CSINode {
	return &storage.CSINode{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: storage.CSINodeSpec{
			Drivers: []storage.CSINodeDriver{
				{
					Name:         "valid-driver-name",
					NodeID:       "valid-node",
					TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					Allocatable:  &storage.VolumeNodeResources{Count: utilpointer.Int32Ptr(10)},
				},
			},
		},
	}
}
