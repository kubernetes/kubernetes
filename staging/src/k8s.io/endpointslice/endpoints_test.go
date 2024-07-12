/*
Copyright 2024 The Kubernetes Authors.

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

package endpointslice

import (
	"reflect"
	"testing"

	corev1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	endpointsliceutil "k8s.io/endpointslice/util"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
)

func TestDesiredEndpointSlicesFromEndpoints(t *testing.T) {
	endpointsSubsets := []corev1.EndpointSubset{
		{
			Addresses: []corev1.EndpointAddress{
				{
					IP:       "172.16.1.1",
					NodeName: ptr.To("worker"),
					TargetRef: &corev1.ObjectReference{
						Kind:      "Pod",
						Namespace: "default",
						Name:      "pod-1",
						UID:       "d77ee8a4-96a8-41e4-b8ed-2880eae70cc5",
					},
				},
			},
			NotReadyAddresses: []corev1.EndpointAddress{
				{
					IP:       "172.16.1.4",
					NodeName: ptr.To("worker"),
					TargetRef: &corev1.ObjectReference{
						Kind:      "Pod",
						Namespace: "default",
						Name:      "pod-4",
						UID:       "d77ee8a4-96a8-41e4-b8ed-2880eae70cc8",
					},
				},
			},
			Ports: []corev1.EndpointPort{
				{
					Name:     "",
					Port:     5000,
					Protocol: corev1.ProtocolTCP,
				},
			},
		},
		{
			Addresses: []corev1.EndpointAddress{
				{
					IP:       "172.16.1.2",
					NodeName: ptr.To("worker"),
					TargetRef: &corev1.ObjectReference{
						Kind:      "Pod",
						Namespace: "default",
						Name:      "pod-2",
						UID:       "d77ee8a4-96a8-41e4-b8ed-2880eae70cc6",
					},
				},
				{
					IP:       "172.16.1.3",
					NodeName: ptr.To("worker"),
					TargetRef: &corev1.ObjectReference{
						Kind:      "Pod",
						Namespace: "default",
						Name:      "pod-3",
						UID:       "d77ee8a4-96a8-41e4-b8ed-2880eae70cc7",
					},
				},
				{
					IP:       "fc00:f853:ccd:e793::3",
					NodeName: ptr.To("worker"),
					TargetRef: &corev1.ObjectReference{
						Kind:      "Pod",
						Namespace: "default",
						Name:      "pod-1",
						UID:       "d77ee8a4-96a8-41e4-b8ed-2880eae70cc5",
					},
				},
			},
			NotReadyAddresses: []corev1.EndpointAddress{},
			Ports: []corev1.EndpointPort{
				{
					Name:     "",
					Port:     4000,
					Protocol: corev1.ProtocolTCP,
				},
				{
					Name:     "App",
					Port:     3000,
					Protocol: corev1.ProtocolTCP,
				},
			},
		},
		{
			Addresses: []corev1.EndpointAddress{},
			NotReadyAddresses: []corev1.EndpointAddress{
				{
					IP:       "172.16.1.2",
					NodeName: ptr.To("worker"),
					TargetRef: &corev1.ObjectReference{
						Kind:      "Pod",
						Namespace: "default",
						Name:      "pod-2",
						UID:       "d77ee8a4-96a8-41e4-b8ed-2880eae70cc6",
					},
				}},
			Ports: []corev1.EndpointPort{
				{
					Name:     "",
					Port:     5000,
					Protocol: corev1.ProtocolTCP,
				},
			},
		},
	}

	endpoint00 := &discovery.Endpoint{
		Addresses:  []string{endpointsSubsets[0].Addresses[0].IP},
		Conditions: discovery.EndpointConditions{Ready: ptr.To(true)},
		NodeName:   endpointsSubsets[0].Addresses[0].NodeName,
		TargetRef:  &corev1.ObjectReference{Kind: "Pod", Namespace: endpointsSubsets[0].Addresses[0].TargetRef.Namespace, Name: endpointsSubsets[0].Addresses[0].TargetRef.Name, UID: endpointsSubsets[0].Addresses[0].TargetRef.UID},
	}
	endpoint01 := &discovery.Endpoint{
		Addresses:  []string{endpointsSubsets[0].NotReadyAddresses[0].IP},
		Conditions: discovery.EndpointConditions{Ready: ptr.To(true)},
		NodeName:   endpointsSubsets[0].NotReadyAddresses[0].NodeName,
		TargetRef:  &corev1.ObjectReference{Kind: "Pod", Namespace: endpointsSubsets[0].NotReadyAddresses[0].TargetRef.Namespace, Name: endpointsSubsets[0].NotReadyAddresses[0].TargetRef.Name, UID: endpointsSubsets[0].NotReadyAddresses[0].TargetRef.UID},
	}
	endpoint10 := &discovery.Endpoint{
		Addresses:  []string{endpointsSubsets[1].Addresses[0].IP},
		Conditions: discovery.EndpointConditions{Ready: ptr.To(true)},
		NodeName:   endpointsSubsets[1].Addresses[0].NodeName,
		TargetRef:  &corev1.ObjectReference{Kind: "Pod", Namespace: endpointsSubsets[1].Addresses[0].TargetRef.Namespace, Name: endpointsSubsets[1].Addresses[0].TargetRef.Name, UID: endpointsSubsets[1].Addresses[0].TargetRef.UID},
	}
	endpoint11 := &discovery.Endpoint{
		Addresses:  []string{endpointsSubsets[1].Addresses[1].IP},
		Conditions: discovery.EndpointConditions{Ready: ptr.To(true)},
		NodeName:   endpointsSubsets[1].Addresses[1].NodeName,
		TargetRef:  &corev1.ObjectReference{Kind: "Pod", Namespace: endpointsSubsets[1].Addresses[1].TargetRef.Namespace, Name: endpointsSubsets[1].Addresses[1].TargetRef.Name, UID: endpointsSubsets[1].Addresses[1].TargetRef.UID},
	}
	endpoint12 := &discovery.Endpoint{
		Addresses:  []string{endpointsSubsets[1].Addresses[2].IP},
		Conditions: discovery.EndpointConditions{Ready: ptr.To(true)},
		NodeName:   endpointsSubsets[1].Addresses[2].NodeName,
		TargetRef:  &corev1.ObjectReference{Kind: "Pod", Namespace: endpointsSubsets[1].Addresses[2].TargetRef.Namespace, Name: endpointsSubsets[1].Addresses[2].TargetRef.Name, UID: endpointsSubsets[1].Addresses[2].TargetRef.UID},
	}
	endpoint20 := &discovery.Endpoint{
		Addresses:  []string{endpointsSubsets[2].NotReadyAddresses[0].IP},
		Conditions: discovery.EndpointConditions{Ready: ptr.To(false)},
		NodeName:   endpointsSubsets[2].NotReadyAddresses[0].NodeName,
		TargetRef:  &corev1.ObjectReference{Kind: "Pod", Namespace: endpointsSubsets[2].NotReadyAddresses[0].TargetRef.Namespace, Name: endpointsSubsets[2].NotReadyAddresses[0].TargetRef.Name, UID: endpointsSubsets[2].NotReadyAddresses[0].TargetRef.UID},
	}

	type args struct {
		endpoints             *corev1.Endpoints
		maxEndpointsPerSubset int32
	}
	tests := []struct {
		name                      string
		args                      args
		wantDesiredEndpointSlices []*EndpointPortAddressType
		wantSupportedAddressTypes sets.Set[discovery.AddressType]
	}{
		{
			name: "empty endpoints",
			args: args{
				endpoints:             &corev1.Endpoints{Subsets: []corev1.EndpointSubset{}},
				maxEndpointsPerSubset: 1,
			},
			wantDesiredEndpointSlices: []*EndpointPortAddressType{},
			wantSupportedAddressTypes: sets.Set[discovery.AddressType]{},
		},
		{
			name: "1 endpoint slice",
			args: args{
				endpoints:             &corev1.Endpoints{Subsets: endpointsSubsets[:1]},
				maxEndpointsPerSubset: 2,
			},
			wantDesiredEndpointSlices: []*EndpointPortAddressType{
				{
					EndpointSet: func() endpointsliceutil.EndpointSet {
						es := endpointsliceutil.EndpointSet{}
						es.Insert(endpoint00, endpoint01)
						return es
					}(),
					Ports: []discovery.EndpointPort{
						{Name: &endpointsSubsets[0].Ports[0].Name, Protocol: &endpointsSubsets[0].Ports[0].Protocol, Port: &endpointsSubsets[0].Ports[0].Port},
					},
					AddressType: discovery.AddressTypeIPv4,
				},
			},
			wantSupportedAddressTypes: sets.Set[discovery.AddressType]{discovery.AddressTypeIPv4: sets.Empty{}},
		},
		{
			name: "endpoint slice limit on non ready (maxEndpointsPerSubset)",
			args: args{
				endpoints:             &corev1.Endpoints{Subsets: endpointsSubsets[:1]},
				maxEndpointsPerSubset: 1,
			},
			wantDesiredEndpointSlices: []*EndpointPortAddressType{
				{
					EndpointSet: func() endpointsliceutil.EndpointSet {
						es := endpointsliceutil.EndpointSet{}
						es.Insert(endpoint00)
						return es
					}(),
					Ports: []discovery.EndpointPort{
						{Name: &endpointsSubsets[0].Ports[0].Name, Protocol: &endpointsSubsets[0].Ports[0].Protocol, Port: &endpointsSubsets[0].Ports[0].Port},
					},
					AddressType: discovery.AddressTypeIPv4,
				},
			},
			wantSupportedAddressTypes: sets.Set[discovery.AddressType]{discovery.AddressTypeIPv4: sets.Empty{}},
		},
		{
			name: "endpoint slice limit on ready (maxEndpointsPerSubset)",
			args: args{
				endpoints:             &corev1.Endpoints{Subsets: endpointsSubsets[1:2]},
				maxEndpointsPerSubset: 1,
			},
			wantDesiredEndpointSlices: []*EndpointPortAddressType{
				{
					EndpointSet: func() endpointsliceutil.EndpointSet {
						es := endpointsliceutil.EndpointSet{}
						es.Insert(endpoint10)
						return es
					}(),
					Ports: []discovery.EndpointPort{
						{Name: &endpointsSubsets[1].Ports[1].Name, Protocol: &endpointsSubsets[1].Ports[1].Protocol, Port: &endpointsSubsets[1].Ports[1].Port},
						{Name: &endpointsSubsets[1].Ports[0].Name, Protocol: &endpointsSubsets[1].Ports[0].Protocol, Port: &endpointsSubsets[1].Ports[0].Port},
					},
					AddressType: discovery.AddressTypeIPv4,
				},
			},
			wantSupportedAddressTypes: sets.Set[discovery.AddressType]{discovery.AddressTypeIPv4: sets.Empty{}},
		},
		{
			name: "3 endpoint slices (dualstack)",
			args: args{
				endpoints:             &corev1.Endpoints{Subsets: endpointsSubsets[:2]},
				maxEndpointsPerSubset: 3,
			},
			wantDesiredEndpointSlices: []*EndpointPortAddressType{
				{
					EndpointSet: func() endpointsliceutil.EndpointSet {
						es := endpointsliceutil.EndpointSet{}
						es.Insert(endpoint00, endpoint01)
						return es
					}(),
					Ports: []discovery.EndpointPort{
						{Name: &endpointsSubsets[0].Ports[0].Name, Protocol: &endpointsSubsets[0].Ports[0].Protocol, Port: &endpointsSubsets[0].Ports[0].Port},
					},
					AddressType: discovery.AddressTypeIPv4,
				},
				{
					EndpointSet: func() endpointsliceutil.EndpointSet {
						es := endpointsliceutil.EndpointSet{}
						es.Insert(endpoint10, endpoint11)
						return es
					}(),
					Ports: []discovery.EndpointPort{
						{Name: &endpointsSubsets[1].Ports[1].Name, Protocol: &endpointsSubsets[1].Ports[1].Protocol, Port: &endpointsSubsets[1].Ports[1].Port},
						{Name: &endpointsSubsets[1].Ports[0].Name, Protocol: &endpointsSubsets[1].Ports[0].Protocol, Port: &endpointsSubsets[1].Ports[0].Port},
					},
					AddressType: discovery.AddressTypeIPv4,
				},
				{
					EndpointSet: func() endpointsliceutil.EndpointSet {
						es := endpointsliceutil.EndpointSet{}
						es.Insert(endpoint12)
						return es
					}(),
					Ports: []discovery.EndpointPort{
						{Name: &endpointsSubsets[1].Ports[1].Name, Protocol: &endpointsSubsets[1].Ports[1].Protocol, Port: &endpointsSubsets[1].Ports[1].Port},
						{Name: &endpointsSubsets[1].Ports[0].Name, Protocol: &endpointsSubsets[1].Ports[0].Protocol, Port: &endpointsSubsets[1].Ports[0].Port},
					},
					AddressType: discovery.AddressTypeIPv6,
				},
			},
			wantSupportedAddressTypes: sets.Set[discovery.AddressType]{discovery.AddressTypeIPv4: sets.Empty{}, discovery.AddressTypeIPv6: sets.Empty{}},
		},
		{
			name: "merge subsets",
			args: args{
				endpoints:             &corev1.Endpoints{Subsets: []corev1.EndpointSubset{endpointsSubsets[0], endpointsSubsets[2]}},
				maxEndpointsPerSubset: 3,
			},
			wantDesiredEndpointSlices: []*EndpointPortAddressType{
				{
					EndpointSet: func() endpointsliceutil.EndpointSet {
						es := endpointsliceutil.EndpointSet{}
						es.Insert(endpoint00, endpoint01, endpoint20)
						return es
					}(),
					Ports: []discovery.EndpointPort{
						{Name: &endpointsSubsets[0].Ports[0].Name, Protocol: &endpointsSubsets[0].Ports[0].Protocol, Port: &endpointsSubsets[0].Ports[0].Port},
					},
					AddressType: discovery.AddressTypeIPv4,
				},
			},
			wantSupportedAddressTypes: sets.Set[discovery.AddressType]{discovery.AddressTypeIPv4: sets.Empty{}},
		},
		{
			name: "only non ready",
			args: args{
				endpoints:             &corev1.Endpoints{Subsets: []corev1.EndpointSubset{endpointsSubsets[2]}},
				maxEndpointsPerSubset: 1,
			},
			wantDesiredEndpointSlices: []*EndpointPortAddressType{
				{
					EndpointSet: func() endpointsliceutil.EndpointSet {
						es := endpointsliceutil.EndpointSet{}
						es.Insert(endpoint20)
						return es
					}(),
					Ports: []discovery.EndpointPort{
						{Name: &endpointsSubsets[0].Ports[0].Name, Protocol: &endpointsSubsets[0].Ports[0].Protocol, Port: &endpointsSubsets[0].Ports[0].Port},
					},
					AddressType: discovery.AddressTypeIPv4,
				},
			},
			wantSupportedAddressTypes: sets.Set[discovery.AddressType]{discovery.AddressTypeIPv4: sets.Empty{}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1 := DesiredEndpointSlicesFromEndpoints(tt.args.endpoints, tt.args.maxEndpointsPerSubset)
			if !CompareEndpointPortAddressTypeSlices(got, tt.wantDesiredEndpointSlices) {
				t.Errorf("DesiredEndpointSlicesFromEndpoints() got (desiredEndpointSlices) = %v, want %v", got, tt.wantDesiredEndpointSlices)
			}
			if !reflect.DeepEqual(got1, tt.wantSupportedAddressTypes) {
				t.Errorf("DesiredEndpointSlicesFromEndpoints() got (supportedAddressTypes) = %v, want %v", got1, tt.wantSupportedAddressTypes)
			}
		})
	}
}

func TestLabelsAnnotationsFromEndpoints_SetLabelsAnnotations(t *testing.T) {
	type args struct {
		logger         klog.Logger
		epSlice        *discovery.EndpointSlice
		controllerName string
	}
	tests := []struct {
		name                               string
		labelabelsAnnotationsFromEndpoints *LabelsAnnotationsFromEndpoints
		args                               args
		wantLabels                         map[string]string
		wantAnnotations                    map[string]string
		wantUpdated                        bool
	}{
		{
			name: "no labels, no annotations to clone",
			labelabelsAnnotationsFromEndpoints: &LabelsAnnotationsFromEndpoints{
				Endpoints: &corev1.Endpoints{
					ObjectMeta: v1.ObjectMeta{
						Name: "endpoints-a",
					},
				},
			},
			args: args{
				epSlice:        &discovery.EndpointSlice{},
				controllerName: "abc",
			},
			wantLabels: map[string]string{
				"endpointslice.kubernetes.io/managed-by": "abc",
				"kubernetes.io/service-name":             "endpoints-a",
			},
			wantAnnotations: map[string]string{},
			wantUpdated:     true,
		},
		{
			name: "no labels, no annotations to clone, wrong mandatory endpointslice labels",
			labelabelsAnnotationsFromEndpoints: &LabelsAnnotationsFromEndpoints{
				Endpoints: &corev1.Endpoints{
					ObjectMeta: v1.ObjectMeta{
						Name: "endpoints-a",
					},
				},
			},
			args: args{
				epSlice: &discovery.EndpointSlice{
					ObjectMeta: v1.ObjectMeta{
						Labels: map[string]string{
							"endpointslice.kubernetes.io/managed-by": "a",
							"kubernetes.io/service-name":             "endpoints-a",
						},
						Annotations: map[string]string{},
					},
				},
				controllerName: "abc",
			},
			wantLabels: map[string]string{
				"endpointslice.kubernetes.io/managed-by": "abc",
				"kubernetes.io/service-name":             "endpoints-a",
			},
			wantAnnotations: map[string]string{},
			wantUpdated:     true,
		},
		{
			name: "no labels, no annotations to clone, correct mandatory endpointslice labels",
			labelabelsAnnotationsFromEndpoints: &LabelsAnnotationsFromEndpoints{
				Endpoints: &corev1.Endpoints{
					ObjectMeta: v1.ObjectMeta{
						Name: "endpoints-a",
					},
				},
			},
			args: args{
				epSlice: &discovery.EndpointSlice{
					ObjectMeta: v1.ObjectMeta{
						Labels: map[string]string{
							"endpointslice.kubernetes.io/managed-by": "abc",
							"kubernetes.io/service-name":             "endpoints-a",
						},
						Annotations: map[string]string{},
					},
				},
				controllerName: "abc",
			},
			wantLabels: map[string]string{
				"endpointslice.kubernetes.io/managed-by": "abc",
				"kubernetes.io/service-name":             "endpoints-a",
			},
			wantAnnotations: map[string]string{},
			wantUpdated:     false,
		},
		{
			name: "labels and annotations to clone",
			labelabelsAnnotationsFromEndpoints: &LabelsAnnotationsFromEndpoints{
				Endpoints: &corev1.Endpoints{
					ObjectMeta: v1.ObjectMeta{
						Name: "endpoints-a",
						Labels: map[string]string{
							"foo": "bar",
						},
						Annotations: map[string]string{
							"bar": "foo",
						},
					},
				},
			},
			args: args{
				epSlice:        &discovery.EndpointSlice{},
				controllerName: "abc",
			},
			wantLabels: map[string]string{
				"endpointslice.kubernetes.io/managed-by": "abc",
				"kubernetes.io/service-name":             "endpoints-a",
				"foo":                                    "bar",
			},
			wantAnnotations: map[string]string{
				"bar": "foo",
			},
			wantUpdated: true,
		},
		{
			name: "labels and annotations to clone with correct mandatory endpointslice labels ",
			labelabelsAnnotationsFromEndpoints: &LabelsAnnotationsFromEndpoints{
				Endpoints: &corev1.Endpoints{
					ObjectMeta: v1.ObjectMeta{
						Name: "endpoints-a",
						Labels: map[string]string{
							"foo": "bar",
						},
						Annotations: map[string]string{
							"bar": "foo",
						},
					},
				},
			},
			args: args{
				epSlice: &discovery.EndpointSlice{
					ObjectMeta: v1.ObjectMeta{
						Labels: map[string]string{
							"endpointslice.kubernetes.io/managed-by": "abc",
							"kubernetes.io/service-name":             "endpoints-a",
						},
						Annotations: map[string]string{},
					},
				},
				controllerName: "abc",
			},
			wantLabels: map[string]string{
				"endpointslice.kubernetes.io/managed-by": "abc",
				"kubernetes.io/service-name":             "endpoints-a",
				"foo":                                    "bar",
			},
			wantAnnotations: map[string]string{
				"bar": "foo",
			},
			wantUpdated: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1, got2 := tt.labelabelsAnnotationsFromEndpoints.SetLabelsAnnotations(tt.args.logger, tt.args.epSlice, tt.args.controllerName)
			if !reflect.DeepEqual(got, tt.wantLabels) {
				t.Errorf("LabelsAnnotationsFromEndpoints.Set() got (labels) = %v, want %v", got, tt.wantLabels)
			}
			if !reflect.DeepEqual(got1, tt.wantAnnotations) {
				t.Errorf("LabelsAnnotationsFromEndpoints.Set() got (annotations) = %v, want %v", got1, tt.wantAnnotations)
			}
			if got2 != tt.wantUpdated {
				t.Errorf("LabelsAnnotationsFromEndpoints.Set() got (updated) = %v, want %v", got2, tt.wantUpdated)
			}
		})
	}
}
