/*
Copyright 2017 The Kubernetes Authors.

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
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/extensions"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
)

func TestV1StatefulSetSpecConversion(t *testing.T) {
	replicas := newInt32(2)
	selector := &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}}
	v1Template := v1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec: v1.PodSpec{
			RestartPolicy:   v1.RestartPolicy("bar"),
			SecurityContext: new(v1.PodSecurityContext),
		},
	}
	apiTemplate := api.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec: api.PodSpec{
			RestartPolicy:   api.RestartPolicy("bar"),
			SecurityContext: new(api.PodSecurityContext),
		},
	}
	testcases := map[string]struct {
		stsSpec1 *apps.StatefulSetSpec
		stsSepc2 *appsv1.StatefulSetSpec
	}{
		"StatefulSetSpec Conversion 1": {
			stsSpec1: &apps.StatefulSetSpec{
				Replicas: *replicas,
				Template: apiTemplate,
			},
			stsSepc2: &appsv1.StatefulSetSpec{
				Replicas: replicas,
				Template: v1Template,
			},
		},
		"StatefulSetSpec Conversion 2": {
			stsSpec1: &apps.StatefulSetSpec{
				Replicas:            *replicas,
				Selector:            selector,
				Template:            apiTemplate,
				ServiceName:         "foo",
				PodManagementPolicy: apps.PodManagementPolicyType("bar"),
			},
			stsSepc2: &appsv1.StatefulSetSpec{
				Replicas:            replicas,
				Selector:            selector,
				Template:            v1Template,
				ServiceName:         "foo",
				PodManagementPolicy: appsv1.PodManagementPolicyType("bar"),
			},
		},
	}

	for k, tc := range testcases {
		// apps -> v1
		internal1 := &appsv1.StatefulSetSpec{}
		if err := api.Scheme.Convert(tc.stsSpec1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from extensions to v1", err)
		}

		if !apiequality.Semantic.DeepEqual(internal1, tc.stsSepc2) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "from extensions to v1", tc.stsSepc2, internal1)
		}

		// v1 -> apps
		internal2 := &apps.StatefulSetSpec{}
		if err := api.Scheme.Convert(tc.stsSepc2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from v1 to extensions", err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.stsSpec1) {
			t.Errorf("%q- %q: expected\n\t%#v, got \n\t%#v", k, "from v1 to extensions", tc.stsSpec1, internal2)
		}
	}
}

func TestV1StatefulSetUpdateStrategyConversion(t *testing.T) {
	partition := newInt32(2)
	v1rollingUpdate := new(appsv1.RollingUpdateStatefulSetStrategy)
	v1rollingUpdate.Partition = partition
	appsrollingUpdate := new(apps.RollingUpdateStatefulSetStrategy)
	appsrollingUpdate.Partition = *partition
	testcases := map[string]struct {
		stsUpdateStrategy1 *apps.StatefulSetUpdateStrategy
		stsUpdateStrategy2 *appsv1.StatefulSetUpdateStrategy
	}{
		"StatefulSetUpdateStrategy Conversion 1": {
			stsUpdateStrategy1: &apps.StatefulSetUpdateStrategy{Type: apps.StatefulSetUpdateStrategyType("foo")},
			stsUpdateStrategy2: &appsv1.StatefulSetUpdateStrategy{Type: appsv1.StatefulSetUpdateStrategyType("foo")},
		},
		"StatefulSetUpdateStrategy Conversion 2": {
			stsUpdateStrategy1: &apps.StatefulSetUpdateStrategy{
				Type:          apps.StatefulSetUpdateStrategyType("foo"),
				RollingUpdate: appsrollingUpdate,
			},
			stsUpdateStrategy2: &appsv1.StatefulSetUpdateStrategy{
				Type:          appsv1.StatefulSetUpdateStrategyType("foo"),
				RollingUpdate: v1rollingUpdate,
			},
		},
	}

	for k, tc := range testcases {
		// apps -> v1
		internal1 := &appsv1.StatefulSetUpdateStrategy{}
		if err := api.Scheme.Convert(tc.stsUpdateStrategy1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", "apps -> v1", k, err)
		}

		if !apiequality.Semantic.DeepEqual(internal1, tc.stsUpdateStrategy2) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", "apps -> v1", k, tc.stsUpdateStrategy2, internal1)
		}

		// v1 -> apps
		internal2 := &apps.StatefulSetUpdateStrategy{}
		if err := api.Scheme.Convert(tc.stsUpdateStrategy2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", "v1 -> apps", k, err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.stsUpdateStrategy1) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", "v1 -> apps", k, tc.stsUpdateStrategy1, internal2)
		}
	}
}

func TestV1StatefulSetStatusConversion(t *testing.T) {
	observedGeneration := new(int64)
	*observedGeneration = 2
	collisionCount := new(int32)
	*collisionCount = 1
	testcases := map[string]struct {
		stsStatus1 *apps.StatefulSetStatus
		stsStatus2 *appsv1.StatefulSetStatus
	}{
		"StatefulSetStatus Conversion 1": {
			stsStatus1: &apps.StatefulSetStatus{
				Replicas:           int32(3),
				ReadyReplicas:      int32(1),
				CurrentReplicas:    int32(3),
				UpdatedReplicas:    int32(3),
				CurrentRevision:    "12345",
				UpdateRevision:     "23456",
				ObservedGeneration: observedGeneration,
			},
			stsStatus2: &appsv1.StatefulSetStatus{
				Replicas:           int32(3),
				ReadyReplicas:      int32(1),
				CurrentReplicas:    int32(3),
				UpdatedReplicas:    int32(3),
				CurrentRevision:    "12345",
				UpdateRevision:     "23456",
				ObservedGeneration: *observedGeneration,
			},
		},
		"StatefulSetStatus Conversion 2": {
			stsStatus1: &apps.StatefulSetStatus{
				ObservedGeneration: observedGeneration,
				Replicas:           int32(3),
				ReadyReplicas:      int32(1),
				CurrentReplicas:    int32(3),
				UpdatedReplicas:    int32(3),
				CurrentRevision:    "12345",
				UpdateRevision:     "23456",
				CollisionCount:     collisionCount,
			},
			stsStatus2: &appsv1.StatefulSetStatus{
				ObservedGeneration: *observedGeneration,
				Replicas:           int32(3),
				ReadyReplicas:      int32(1),
				CurrentReplicas:    int32(3),
				UpdatedReplicas:    int32(3),
				CurrentRevision:    "12345",
				UpdateRevision:     "23456",
				CollisionCount:     collisionCount,
			},
		},
	}

	for k, tc := range testcases {
		// apps -> v1
		internal1 := &appsv1.StatefulSetStatus{}
		if err := api.Scheme.Convert(tc.stsStatus1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from apps to v1", err)
		}

		if !apiequality.Semantic.DeepEqual(internal1, tc.stsStatus2) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "from apps to v1", tc.stsStatus2, internal1)
		}

		// v1 -> apps
		internal2 := &apps.StatefulSetStatus{}
		if err := api.Scheme.Convert(tc.stsStatus2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from v1 to apps", err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.stsStatus1) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "from v1 to apps", tc.stsStatus1, internal2)
		}
	}
}

func TestV1RollingUpdateDaemonSetConversion(t *testing.T) {
	intorstr := intstr.FromInt(1)
	testcases := map[string]struct {
		rollingUpdateDs1 *extensions.RollingUpdateDaemonSet
		rollingUpdateDs2 *appsv1.RollingUpdateDaemonSet
	}{
		"RollingUpdateDaemonSet Conversion 2": {
			rollingUpdateDs1: &extensions.RollingUpdateDaemonSet{MaxUnavailable: intorstr},
			rollingUpdateDs2: &appsv1.RollingUpdateDaemonSet{MaxUnavailable: &intorstr},
		},
	}

	for k, tc := range testcases {
		// extensions -> v1
		internal1 := &appsv1.RollingUpdateDaemonSet{}
		if err := api.Scheme.Convert(tc.rollingUpdateDs1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from extensions to v1", err)
		}
		if !apiequality.Semantic.DeepEqual(internal1, tc.rollingUpdateDs2) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "from extensions to v1", tc.rollingUpdateDs2, internal1)
		}

		// v1 -> extensions
		internal2 := &extensions.RollingUpdateDaemonSet{}
		if err := api.Scheme.Convert(tc.rollingUpdateDs2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from v1 to extensions", err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.rollingUpdateDs1) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "from v1 to extensions", tc.rollingUpdateDs1, internal2)
		}
	}
}
