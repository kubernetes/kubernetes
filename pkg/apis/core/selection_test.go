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

package core

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestEventGetAttrs(t *testing.T) {
	eventA := &Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "f0118",
			Namespace: "default",
		},
		InvolvedObject: ObjectReference{
			Kind:            "Pod",
			Name:            "foo",
			Namespace:       "baz",
			UID:             "long uid string",
			APIVersion:      "v1",
			ResourceVersion: "0",
			FieldPath:       "",
		},
		Reason: "ForTesting",
		Source: EventSource{Component: "test"},
		Type:   EventTypeNormal,
	}
	field := EventToSelectableFields(eventA)
	expectA := fields.Set{
		"metadata.name":                  "f0118",
		"metadata.namespace":             "default",
		"involvedObject.kind":            "Pod",
		"involvedObject.name":            "foo",
		"involvedObject.namespace":       "baz",
		"involvedObject.uid":             "long uid string",
		"involvedObject.apiVersion":      "v1",
		"involvedObject.resourceVersion": "0",
		"involvedObject.fieldPath":       "",
		"reason":                         "ForTesting",
		"reportingComponent":             "",
		"source":                         "test",
		"type":                           EventTypeNormal,
	}
	if e, a := expectA, field; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", cmp.Diff(e, a))
	}

	eventB := &Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "f0118",
			Namespace: "default",
		},
		InvolvedObject: ObjectReference{
			Kind:            "Pod",
			Name:            "foo",
			Namespace:       "baz",
			UID:             "long uid string",
			APIVersion:      "v1",
			ResourceVersion: "0",
			FieldPath:       "",
		},
		Reason:              "ForTesting",
		ReportingController: "test",
		Type:                EventTypeNormal,
	}
	field = EventToSelectableFields(eventB)
	expectB := fields.Set{
		"metadata.name":                  "f0118",
		"metadata.namespace":             "default",
		"involvedObject.kind":            "Pod",
		"involvedObject.name":            "foo",
		"involvedObject.namespace":       "baz",
		"involvedObject.uid":             "long uid string",
		"involvedObject.apiVersion":      "v1",
		"involvedObject.resourceVersion": "0",
		"involvedObject.fieldPath":       "",
		"reason":                         "ForTesting",
		"reportingComponent":             "test",
		"source":                         "test",
		"type":                           EventTypeNormal,
	}
	if e, a := expectB, field; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", cmp.Diff(e, a))
	}
}

func TestNodeMatcher(t *testing.T) {
	testFieldMap := map[bool][]fields.Set{
		true: {
			{"metadata.name": "foo"},
		},
		false: {
			{"foo": "bar"},
		},
	}

	for expectedResult, fieldSet := range testFieldMap {
		for _, field := range fieldSet {
			m := NodeMatcher(runtime.Selectors{
				Labels: labels.Everything(),
				Fields: field.AsSelector(),
			})
			_, matchesSingle := m.MatchesSingle()
			if e, a := expectedResult, matchesSingle; e != a {
				t.Errorf("%+v: expected %v, got %v", fieldSet, e, a)
			}
		}
	}
}

func TestPodMatcher(t *testing.T) {
	testCases := []struct {
		in            *Pod
		fieldSelector fields.Selector
		expectMatch   bool
	}{
		{
			in: &Pod{
				Spec: PodSpec{NodeName: "nodeA"},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.nodeName=nodeA"),
			expectMatch:   true,
		},
		{
			in: &Pod{
				Spec: PodSpec{NodeName: "nodeB"},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.nodeName=nodeA"),
			expectMatch:   false,
		},
		{
			in: &Pod{
				Spec: PodSpec{RestartPolicy: RestartPolicyAlways},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.restartPolicy=Always"),
			expectMatch:   true,
		},
		{
			in: &Pod{
				Spec: PodSpec{RestartPolicy: RestartPolicyAlways},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.restartPolicy=Never"),
			expectMatch:   false,
		},
		{
			in: &Pod{
				Spec: PodSpec{SchedulerName: "scheduler1"},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.schedulerName=scheduler1"),
			expectMatch:   true,
		},
		{
			in: &Pod{
				Spec: PodSpec{SchedulerName: "scheduler1"},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.schedulerName=scheduler2"),
			expectMatch:   false,
		},
		{
			in: &Pod{
				Spec: PodSpec{ServiceAccountName: "serviceAccount1"},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.serviceAccountName=serviceAccount1"),
			expectMatch:   true,
		},
		{
			in: &Pod{
				Spec: PodSpec{SchedulerName: "serviceAccount1"},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.serviceAccountName=serviceAccount2"),
			expectMatch:   false,
		},
		{
			in: &Pod{
				Status: PodStatus{Phase: PodRunning},
			},
			fieldSelector: fields.ParseSelectorOrDie("status.phase=Running"),
			expectMatch:   true,
		},
		{
			in: &Pod{
				Status: PodStatus{Phase: PodRunning},
			},
			fieldSelector: fields.ParseSelectorOrDie("status.phase=Pending"),
			expectMatch:   false,
		},
		{
			in: &Pod{
				Status: PodStatus{
					PodIPs: []PodIP{
						{IP: "1.2.3.4"},
					},
				},
			},
			fieldSelector: fields.ParseSelectorOrDie("status.podIP=1.2.3.4"),
			expectMatch:   true,
		},
		{
			in: &Pod{
				Status: PodStatus{
					PodIPs: []PodIP{
						{IP: "1.2.3.4"},
					},
				},
			},
			fieldSelector: fields.ParseSelectorOrDie("status.podIP=4.3.2.1"),
			expectMatch:   false,
		},
		{
			in: &Pod{
				Status: PodStatus{NominatedNodeName: "node1"},
			},
			fieldSelector: fields.ParseSelectorOrDie("status.nominatedNodeName=node1"),
			expectMatch:   true,
		},
		{
			in: &Pod{
				Status: PodStatus{NominatedNodeName: "node1"},
			},
			fieldSelector: fields.ParseSelectorOrDie("status.nominatedNodeName=node2"),
			expectMatch:   false,
		},
		{
			in: &Pod{
				Status: PodStatus{
					PodIPs: []PodIP{
						{IP: "2001:db8::"},
					},
				},
			},
			fieldSelector: fields.ParseSelectorOrDie("status.podIP=2001:db8::"),
			expectMatch:   true,
		},
		{
			in: &Pod{
				Status: PodStatus{
					PodIPs: []PodIP{
						{IP: "2001:db8::"},
					},
				},
			},
			fieldSelector: fields.ParseSelectorOrDie("status.podIP=2001:db7::"),
			expectMatch:   false,
		},
		{
			in: &Pod{
				Spec: PodSpec{
					SecurityContext: &PodSecurityContext{
						HostNetwork: true,
					},
				},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.hostNetwork=true"),
			expectMatch:   true,
		},
		{
			in: &Pod{
				Spec: PodSpec{
					SecurityContext: &PodSecurityContext{
						HostNetwork: true,
					},
				},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.hostNetwork=false"),
			expectMatch:   false,
		},
		{
			in: &Pod{
				Spec: PodSpec{
					SecurityContext: &PodSecurityContext{
						HostNetwork: false,
					},
				},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.hostNetwork=false"),
			expectMatch:   true,
		},
		{
			in: &Pod{
				Spec: PodSpec{},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.hostNetwork=false"),
			expectMatch:   true,
		},
		{
			in: &Pod{
				Spec: PodSpec{},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.hostNetwork=true"),
			expectMatch:   false,
		},
	}
	for _, testCase := range testCases {
		m := PodMatcher(runtime.Selectors{
			Labels: labels.Everything(),
			Fields: testCase.fieldSelector,
		})
		result, err := m.Matches(testCase.in)
		if err != nil {
			t.Errorf("Unexpected error %v", err)
		}
		if result != testCase.expectMatch {
			t.Errorf("Result %v, Expected %v, Selector: %v, Pod: %v", result, testCase.expectMatch, testCase.fieldSelector.String(), testCase.in)
		}
	}
}

func TestServiceMatcher(t *testing.T) {
	testCases := []struct {
		name          string
		in            *Service
		fieldSelector fields.Selector
		expectMatch   bool
	}{
		{
			name: "match on name",
			in: &Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: "testns",
				},
				Spec: ServiceSpec{ClusterIP: ClusterIPNone},
			},
			fieldSelector: fields.ParseSelectorOrDie("metadata.name=test"),
			expectMatch:   true,
		},
		{
			name: "match on namespace",
			in: &Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: "testns",
				},
				Spec: ServiceSpec{ClusterIP: ClusterIPNone},
			},
			fieldSelector: fields.ParseSelectorOrDie("metadata.namespace=testns"),
			expectMatch:   true,
		},
		{
			name: "no match on name",
			in: &Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: "testns",
				},
				Spec: ServiceSpec{ClusterIP: ClusterIPNone},
			},
			fieldSelector: fields.ParseSelectorOrDie("metadata.name=nomatch"),
			expectMatch:   false,
		},
		{
			name: "no match on namespace",
			in: &Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: "testns",
				},
				Spec: ServiceSpec{ClusterIP: ClusterIPNone},
			},
			fieldSelector: fields.ParseSelectorOrDie("metadata.namespace=nomatch"),
			expectMatch:   false,
		},
		{
			name: "match on loadbalancer type service",
			in: &Service{
				Spec: ServiceSpec{Type: ServiceTypeLoadBalancer},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.type=LoadBalancer"),
			expectMatch:   true,
		},
		{
			name: "no match on nodeport type service",
			in: &Service{
				Spec: ServiceSpec{Type: ServiceTypeNodePort},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.type=LoadBalancer"),
			expectMatch:   false,
		},
		{
			name: "match on headless service",
			in: &Service{
				Spec: ServiceSpec{ClusterIP: ClusterIPNone},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.clusterIP=None"),
			expectMatch:   true,
		},
		{
			name: "no match on clusterIP service",
			in: &Service{
				Spec: ServiceSpec{ClusterIP: "192.168.1.1"},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.clusterIP=None"),
			expectMatch:   false,
		},
		{
			name: "match on clusterIP service",
			in: &Service{
				Spec: ServiceSpec{ClusterIP: "192.168.1.1"},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.clusterIP=192.168.1.1"),
			expectMatch:   true,
		},
		{
			name: "match on non-headless service",
			in: &Service{
				Spec: ServiceSpec{ClusterIP: "192.168.1.1"},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.clusterIP!=None"),
			expectMatch:   true,
		},
		{
			name: "match on any ClusterIP set service",
			in: &Service{
				Spec: ServiceSpec{ClusterIP: "192.168.1.1"},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.clusterIP!=\"\""),
			expectMatch:   true,
		},
		{
			name: "match on clusterIP IPv6 service",
			in: &Service{
				Spec: ServiceSpec{ClusterIP: "2001:db2::1"},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.clusterIP=2001:db2::1"),
			expectMatch:   true,
		},
		{
			name: "no match on headless service",
			in: &Service{
				Spec: ServiceSpec{ClusterIP: ClusterIPNone},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.clusterIP=192.168.1.1"),
			expectMatch:   false,
		},
		{
			name: "no match on headless service",
			in: &Service{
				Spec: ServiceSpec{ClusterIP: ClusterIPNone},
			},
			fieldSelector: fields.ParseSelectorOrDie("spec.clusterIP=2001:db2::1"),
			expectMatch:   false,
		},
		{
			name:          "no match on empty service",
			in:            &Service{},
			fieldSelector: fields.ParseSelectorOrDie("spec.clusterIP=None"),
			expectMatch:   false,
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			m := ServiceMatcher(runtime.Selectors{
				Labels: labels.Everything(),
				Fields: testCase.fieldSelector,
			})
			result, err := m.Matches(testCase.in)
			if err != nil {
				t.Errorf("Unexpected error %v", err)
			}
			if result != testCase.expectMatch {
				t.Errorf("Result %v, Expected %v, Selector: %v, Service: %v", result, testCase.expectMatch, testCase.fieldSelector.String(), testCase.in)
			}
		})
	}
}
