/*
Copyright 2014 The Kubernetes Authors.

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

package validation

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	_ "k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/core"
)

func TestValidateNode(t *testing.T) {
	validSelector := map[string]string{"a": "b"}
	invalidSelector := map[string]string{"NoUppercaseOrSpecialCharsLike=Equals": "b"}
	successCases := []core.Node{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "abc",
				Labels: validSelector,
			},
			Status: core.NodeStatus{
				Addresses: []core.NodeAddress{
					{Type: core.NodeExternalIP, Address: "something"},
				},
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
					core.ResourceName("my.org/gpu"):        resource.MustParse("10"),
					core.ResourceName("hugepages-2Mi"):     resource.MustParse("10Gi"),
					core.ResourceName("hugepages-1Gi"):     resource.MustParse("0"),
				},
			},
			Spec: core.NodeSpec{
				ExternalID: "external",
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "abc",
			},
			Status: core.NodeStatus{
				Addresses: []core.NodeAddress{
					{Type: core.NodeExternalIP, Address: "something"},
				},
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("0"),
				},
			},
			Spec: core.NodeSpec{
				ExternalID: "external",
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "dedicated-node1",
			},
			Status: core.NodeStatus{
				Addresses: []core.NodeAddress{
					{Type: core.NodeExternalIP, Address: "something"},
				},
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("0"),
				},
			},
			Spec: core.NodeSpec{
				ExternalID: "external",
				// Add a valid taint to a node
				Taints: []core.Taint{{Key: "GPU", Value: "true", Effect: "NoSchedule"}},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "abc",
				Annotations: map[string]string{
					core.PreferAvoidPodsAnnotationKey: `
							{
							    "preferAvoidPods": [
							        {
							            "podSignature": {
							                "podController": {
							                    "apiVersion": "v1",
							                    "kind": "ReplicationController",
							                    "name": "foo",
							                    "uid": "abcdef123456",
							                    "controller": true
							                }
							            },
							            "reason": "some reason",
							            "message": "some message"
							        }
							    ]
							}`,
				},
			},
			Status: core.NodeStatus{
				Addresses: []core.NodeAddress{
					{Type: core.NodeExternalIP, Address: "something"},
				},
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("0"),
				},
			},
			Spec: core.NodeSpec{
				ExternalID: "external",
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "abc",
			},
			Status: core.NodeStatus{
				Addresses: []core.NodeAddress{
					{Type: core.NodeExternalIP, Address: "something"},
				},
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("0"),
				},
			},
			Spec: core.NodeSpec{
				ExternalID: "external",
				PodCIDR:    "192.168.0.0/16",
			},
		},
	}
	for _, successCase := range successCases {
		if errs := ValidateNode(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := map[string]core.Node{
		"zero-length Name": {
			ObjectMeta: metav1.ObjectMeta{
				Name:   "",
				Labels: validSelector,
			},
			Status: core.NodeStatus{
				Addresses: []core.NodeAddress{},
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
				},
			},
			Spec: core.NodeSpec{
				ExternalID: "external",
			},
		},
		"invalid-labels": {
			ObjectMeta: metav1.ObjectMeta{
				Name:   "abc-123",
				Labels: invalidSelector,
			},
			Status: core.NodeStatus{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
				},
			},
			Spec: core.NodeSpec{
				ExternalID: "external",
			},
		},
		"missing-external-id": {
			ObjectMeta: metav1.ObjectMeta{
				Name:   "abc-123",
				Labels: validSelector,
			},
			Status: core.NodeStatus{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
				},
			},
		},
		"missing-taint-key": {
			ObjectMeta: metav1.ObjectMeta{
				Name: "dedicated-node1",
			},
			Spec: core.NodeSpec{
				ExternalID: "external",
				// Add a taint with an empty key to a node
				Taints: []core.Taint{{Key: "", Value: "special-user-1", Effect: "NoSchedule"}},
			},
		},
		"bad-taint-key": {
			ObjectMeta: metav1.ObjectMeta{
				Name: "dedicated-node1",
			},
			Spec: core.NodeSpec{
				ExternalID: "external",
				// Add a taint with an invalid  key to a node
				Taints: []core.Taint{{Key: "NoUppercaseOrSpecialCharsLike=Equals", Value: "special-user-1", Effect: "NoSchedule"}},
			},
		},
		"bad-taint-value": {
			ObjectMeta: metav1.ObjectMeta{
				Name: "dedicated-node2",
			},
			Status: core.NodeStatus{
				Addresses: []core.NodeAddress{
					{Type: core.NodeExternalIP, Address: "something"},
				},
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("0"),
				},
			},
			Spec: core.NodeSpec{
				ExternalID: "external",
				// Add a taint with a bad value to a node
				Taints: []core.Taint{{Key: "dedicated", Value: "some\\bad\\value", Effect: "NoSchedule"}},
			},
		},
		"missing-taint-effect": {
			ObjectMeta: metav1.ObjectMeta{
				Name: "dedicated-node3",
			},
			Status: core.NodeStatus{
				Addresses: []core.NodeAddress{
					{Type: core.NodeExternalIP, Address: "something"},
				},
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("0"),
				},
			},
			Spec: core.NodeSpec{
				ExternalID: "external",
				// Add a taint with an empty effect to a node
				Taints: []core.Taint{{Key: "dedicated", Value: "special-user-3", Effect: ""}},
			},
		},
		"invalid-taint-effect": {
			ObjectMeta: metav1.ObjectMeta{
				Name: "dedicated-node3",
			},
			Status: core.NodeStatus{
				Addresses: []core.NodeAddress{
					{Type: core.NodeExternalIP, Address: "something"},
				},
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("0"),
				},
			},
			Spec: core.NodeSpec{
				ExternalID: "external",
				// Add a taint with NoExecute effect to a node
				Taints: []core.Taint{{Key: "dedicated", Value: "special-user-3", Effect: "NoScheduleNoAdmit"}},
			},
		},
		"duplicated-taints-with-same-key-effect": {
			ObjectMeta: metav1.ObjectMeta{
				Name: "dedicated-node1",
			},
			Spec: core.NodeSpec{
				ExternalID: "external",
				// Add two taints to the node with the same key and effect; should be rejected.
				Taints: []core.Taint{
					{Key: "dedicated", Value: "special-user-1", Effect: "NoSchedule"},
					{Key: "dedicated", Value: "special-user-2", Effect: "NoSchedule"},
				},
			},
		},
		"missing-podSignature": {
			ObjectMeta: metav1.ObjectMeta{
				Name: "abc-123",
				Annotations: map[string]string{
					core.PreferAvoidPodsAnnotationKey: `
							{
							    "preferAvoidPods": [
							        {
							            "reason": "some reason",
							            "message": "some message"
							        }
							    ]
							}`,
				},
			},
			Status: core.NodeStatus{
				Addresses: []core.NodeAddress{},
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("0"),
				},
			},
			Spec: core.NodeSpec{
				ExternalID: "external",
			},
		},
		"invalid-podController": {
			ObjectMeta: metav1.ObjectMeta{
				Name: "abc-123",
				Annotations: map[string]string{
					core.PreferAvoidPodsAnnotationKey: `
							{
							    "preferAvoidPods": [
							        {
							            "podSignature": {
							                "podController": {
							                    "apiVersion": "v1",
							                    "kind": "ReplicationController",
							                    "name": "foo",
                                                                           "uid": "abcdef123456",
                                                                           "controller": false
							                }
							            },
							            "reason": "some reason",
							            "message": "some message"
							        }
							    ]
							}`,
				},
			},
			Status: core.NodeStatus{
				Addresses: []core.NodeAddress{},
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("0"),
				},
			},
			Spec: core.NodeSpec{
				ExternalID: "external",
			},
		},
		"multiple-pre-allocated-hugepages": {
			ObjectMeta: metav1.ObjectMeta{
				Name:   "abc",
				Labels: validSelector,
			},
			Status: core.NodeStatus{
				Addresses: []core.NodeAddress{
					{Type: core.NodeExternalIP, Address: "something"},
				},
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
					core.ResourceName("my.org/gpu"):        resource.MustParse("10"),
					core.ResourceName("hugepages-2Mi"):     resource.MustParse("10Gi"),
					core.ResourceName("hugepages-1Gi"):     resource.MustParse("10Gi"),
				},
			},
			Spec: core.NodeSpec{
				ExternalID: "external",
			},
		},
		"invalid-pod-cidr": {
			ObjectMeta: metav1.ObjectMeta{
				Name: "abc",
			},
			Status: core.NodeStatus{
				Addresses: []core.NodeAddress{
					{Type: core.NodeExternalIP, Address: "something"},
				},
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("0"),
				},
			},
			Spec: core.NodeSpec{
				ExternalID: "external",
				PodCIDR:    "192.168.0.0",
			},
		},
	}
	for k, v := range errorCases {
		errs := ValidateNode(&v)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
		for i := range errs {
			field := errs[i].Field
			expectedFields := map[string]bool{
				"metadata.name":                                                                                               true,
				"metadata.labels":                                                                                             true,
				"metadata.annotations":                                                                                        true,
				"metadata.namespace":                                                                                          true,
				"spec.externalID":                                                                                             true,
				"spec.taints[0].key":                                                                                          true,
				"spec.taints[0].value":                                                                                        true,
				"spec.taints[0].effect":                                                                                       true,
				"metadata.annotations.scheduler.alpha.kubernetes.io/preferAvoidPods[0].PodSignature":                          true,
				"metadata.annotations.scheduler.alpha.kubernetes.io/preferAvoidPods[0].PodSignature.PodController.Controller": true,
			}
			if val, ok := expectedFields[field]; ok {
				if !val {
					t.Errorf("%s: missing prefix for: %v", k, errs[i])
				}
			}
		}
	}
}

func TestValidateNodeUpdate(t *testing.T) {
	tests := []struct {
		oldNode core.Node
		node    core.Node
		valid   bool
	}{
		{core.Node{}, core.Node{}, true},
		{core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo"}},
			core.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "bar"},
			}, false},
		{core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"foo": "bar"},
			},
		}, core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"foo": "baz"},
			},
		}, true},
		{core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
			},
		}, core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"foo": "baz"},
			},
		}, true},
		{core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"bar": "foo"},
			},
		}, core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"foo": "baz"},
			},
		}, true},
		{core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
			},
			Spec: core.NodeSpec{
				PodCIDR: "",
			},
		}, core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
			},
			Spec: core.NodeSpec{
				PodCIDR: "192.168.0.0/16",
			},
		}, true},
		{core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
			},
			Spec: core.NodeSpec{
				PodCIDR: "192.123.0.0/16",
			},
		}, core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
			},
			Spec: core.NodeSpec{
				PodCIDR: "192.168.0.0/16",
			},
		}, false},
		{core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
			},
			Status: core.NodeStatus{
				Capacity: core.ResourceList{
					core.ResourceCPU:    resource.MustParse("10000"),
					core.ResourceMemory: resource.MustParse("100"),
				},
			},
		}, core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
			},
			Status: core.NodeStatus{
				Capacity: core.ResourceList{
					core.ResourceCPU:    resource.MustParse("100"),
					core.ResourceMemory: resource.MustParse("10000"),
				},
			},
		}, true},
		{core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"bar": "foo"},
			},
			Status: core.NodeStatus{
				Capacity: core.ResourceList{
					core.ResourceCPU:    resource.MustParse("10000"),
					core.ResourceMemory: resource.MustParse("100"),
				},
			},
		}, core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"bar": "fooobaz"},
			},
			Status: core.NodeStatus{
				Capacity: core.ResourceList{
					core.ResourceCPU:    resource.MustParse("100"),
					core.ResourceMemory: resource.MustParse("10000"),
				},
			},
		}, true},
		{core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"bar": "foo"},
			},
			Status: core.NodeStatus{
				Addresses: []core.NodeAddress{
					{Type: core.NodeExternalIP, Address: "1.2.3.4"},
				},
			},
		}, core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"bar": "fooobaz"},
			},
		}, true},
		{core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"foo": "baz"},
			},
		}, core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"Foo": "baz"},
			},
		}, true},
		{core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
			},
			Spec: core.NodeSpec{
				Unschedulable: false,
			},
		}, core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
			},
			Spec: core.NodeSpec{
				Unschedulable: true,
			},
		}, true},
		{core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
			},
			Spec: core.NodeSpec{
				Unschedulable: false,
			},
		}, core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
			},
			Status: core.NodeStatus{
				Addresses: []core.NodeAddress{
					{Type: core.NodeExternalIP, Address: "1.1.1.1"},
					{Type: core.NodeExternalIP, Address: "1.1.1.1"},
				},
			},
		}, false},
		{core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
			},
			Spec: core.NodeSpec{
				Unschedulable: false,
			},
		}, core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
			},
			Status: core.NodeStatus{
				Addresses: []core.NodeAddress{
					{Type: core.NodeExternalIP, Address: "1.1.1.1"},
					{Type: core.NodeInternalIP, Address: "10.1.1.1"},
				},
			},
		}, true},
		{core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
			},
		}, core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
				Annotations: map[string]string{
					core.PreferAvoidPodsAnnotationKey: `
							{
							    "preferAvoidPods": [
							        {
							            "podSignature": {
							                "podController": {
							                    "apiVersion": "v1",
							                    "kind": "ReplicationController",
							                    "name": "foo",
                                                                           "uid": "abcdef123456",
                                                                           "controller": true
							                }
							            },
							            "reason": "some reason",
							            "message": "some message"
							        }
							    ]
							}`,
				},
			},
			Spec: core.NodeSpec{
				Unschedulable: false,
			},
		}, true},
		{core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
			},
		}, core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
				Annotations: map[string]string{
					core.PreferAvoidPodsAnnotationKey: `
							{
							    "preferAvoidPods": [
							        {
							            "reason": "some reason",
							            "message": "some message"
							        }
							    ]
							}`,
				},
			},
		}, false},
		{core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
			},
		}, core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
				Annotations: map[string]string{
					core.PreferAvoidPodsAnnotationKey: `
							{
							    "preferAvoidPods": [
							        {
							            "podSignature": {
							                "podController": {
							                    "apiVersion": "v1",
							                    "kind": "ReplicationController",
							                    "name": "foo",
							                    "uid": "abcdef123456",
							                    "controller": false
							                }
							            },
							            "reason": "some reason",
							            "message": "some message"
							        }
							    ]
							}`,
				},
			},
		}, false},
		{core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "valid-extended-resources",
			},
		}, core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "valid-extended-resources",
			},
			Status: core.NodeStatus{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
					core.ResourceName("example.com/a"):     resource.MustParse("5"),
					core.ResourceName("example.com/b"):     resource.MustParse("10"),
				},
			},
		}, true},
		{core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "invalid-fractional-extended-capacity",
			},
		}, core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "invalid-fractional-extended-capacity",
			},
			Status: core.NodeStatus{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
					core.ResourceName("example.com/a"):     resource.MustParse("500m"),
				},
			},
		}, false},
		{core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "invalid-fractional-extended-allocatable",
			},
		}, core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "invalid-fractional-extended-allocatable",
			},
			Status: core.NodeStatus{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
					core.ResourceName("example.com/a"):     resource.MustParse("5"),
				},
				Allocatable: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
					core.ResourceName("example.com/a"):     resource.MustParse("4.5"),
				},
			},
		}, false},
		{core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "update-provider-id-when-not-set",
			},
		}, core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "update-provider-id-when-not-set",
			},
			Spec: core.NodeSpec{
				ProviderID: "provider:///new",
			},
		}, true},
		{core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "update-provider-id-when-set",
			},
			Spec: core.NodeSpec{
				ProviderID: "provider:///old",
			},
		}, core.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "update-provider-id-when-set",
			},
			Spec: core.NodeSpec{
				ProviderID: "provider:///new",
			},
		}, false},
	}
	for i, test := range tests {
		test.oldNode.ObjectMeta.ResourceVersion = "1"
		test.node.ObjectMeta.ResourceVersion = "1"
		errs := ValidateNodeUpdate(&test.node, &test.oldNode)
		if test.valid && len(errs) > 0 {
			t.Errorf("%d: Unexpected error: %v", i, errs)
			t.Logf("%#v vs %#v", test.oldNode.ObjectMeta, test.node.ObjectMeta)
		}
		if !test.valid && len(errs) == 0 {
			t.Errorf("%d: Unexpected non-error", i)
		}
	}
}
