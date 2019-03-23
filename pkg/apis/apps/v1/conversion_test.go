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
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/apps"
	api "k8s.io/kubernetes/pkg/apis/core"
	utilpointer "k8s.io/utils/pointer"
)

func TestV12StatefulSetSpecConversion(t *testing.T) {
	replicas := utilpointer.Int32Ptr(2)
	selector := &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}}
	appsv1Template := v1.PodTemplateSpec{
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
				Template: appsv1Template,
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
				Template:            appsv1Template,
				ServiceName:         "foo",
				PodManagementPolicy: appsv1.PodManagementPolicyType("bar"),
			},
		},
	}

	for k, tc := range testcases {
		// apps -> appsv1
		internal1 := &appsv1.StatefulSetSpec{}
		if err := legacyscheme.Scheme.Convert(tc.stsSpec1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from apps to appsv1", err)
		}

		if !apiequality.Semantic.DeepEqual(internal1, tc.stsSepc2) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "from apps to appsv1", tc.stsSepc2, internal1)
		}

		// appsv1 -> apps
		internal2 := &apps.StatefulSetSpec{}
		if err := legacyscheme.Scheme.Convert(tc.stsSepc2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from appsv1 to apps", err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.stsSpec1) {
			t.Errorf("%q- %q: expected\n\t%#v, got \n\t%#v", k, "from appsv1 to apps", tc.stsSpec1, internal2)
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
		// apps -> appsv1
		internal1 := &appsv1.StatefulSetStatus{}
		if err := legacyscheme.Scheme.Convert(tc.stsStatus1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from apps to appsv1", err)
		}

		if !apiequality.Semantic.DeepEqual(internal1, tc.stsStatus2) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "from apps to appsv1", tc.stsStatus2, internal1)
		}

		// appsv1 -> apps
		internal2 := &apps.StatefulSetStatus{}
		if err := legacyscheme.Scheme.Convert(tc.stsStatus2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from appsv1 to apps", err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.stsStatus1) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "from appsv1 to apps", tc.stsStatus1, internal2)
		}
	}
}

func TestV1StatefulSetUpdateStrategyConversion(t *testing.T) {
	partition := utilpointer.Int32Ptr(2)
	appsv1rollingUpdate := new(appsv1.RollingUpdateStatefulSetStrategy)
	appsv1rollingUpdate.Partition = partition
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
				RollingUpdate: appsv1rollingUpdate,
			},
		},
	}

	for k, tc := range testcases {
		// apps -> appsv1
		internal1 := &appsv1.StatefulSetUpdateStrategy{}
		if err := legacyscheme.Scheme.Convert(tc.stsUpdateStrategy1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", "apps -> appsv1", k, err)
		}

		if !apiequality.Semantic.DeepEqual(internal1, tc.stsUpdateStrategy2) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", "apps -> appsv1", k, tc.stsUpdateStrategy2, internal1)
		}

		// appsv1 -> apps
		internal2 := &apps.StatefulSetUpdateStrategy{}
		if err := legacyscheme.Scheme.Convert(tc.stsUpdateStrategy2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", "appsv1 -> apps", k, err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.stsUpdateStrategy1) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", "appsv1 -> apps", k, tc.stsUpdateStrategy1, internal2)
		}
	}
}

func TestV1RollingUpdateDaemonSetConversion(t *testing.T) {
	intorstr := intstr.FromInt(1)
	testcases := map[string]struct {
		rollingUpdateDs1 *apps.RollingUpdateDaemonSet
		rollingUpdateDs2 *appsv1.RollingUpdateDaemonSet
	}{
		"RollingUpdateDaemonSet Conversion 2": {
			rollingUpdateDs1: &apps.RollingUpdateDaemonSet{MaxUnavailable: intorstr},
			rollingUpdateDs2: &appsv1.RollingUpdateDaemonSet{MaxUnavailable: &intorstr},
		},
	}

	for k, tc := range testcases {
		// apps -> v1
		internal1 := &appsv1.RollingUpdateDaemonSet{}
		if err := legacyscheme.Scheme.Convert(tc.rollingUpdateDs1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from apps to v1", err)
		}
		if !apiequality.Semantic.DeepEqual(internal1, tc.rollingUpdateDs2) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "from apps to v1", tc.rollingUpdateDs2, internal1)
		}

		// v1 -> apps
		internal2 := &apps.RollingUpdateDaemonSet{}
		if err := legacyscheme.Scheme.Convert(tc.rollingUpdateDs2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from v1 to apps", err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.rollingUpdateDs1) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "from v1 to apps", tc.rollingUpdateDs1, internal2)
		}
	}
}

func TestV1DeploymentConversion(t *testing.T) {
	replica := utilpointer.Int32Ptr(2)
	rollbackTo := new(apps.RollbackConfig)
	rollbackTo.Revision = int64(2)
	testcases := map[string]struct {
		deployment1 *apps.Deployment
		deployment2 *appsv1.Deployment
	}{
		"Deployment Conversion 1": {
			deployment1: &apps.Deployment{
				Spec: apps.DeploymentSpec{
					Replicas:   *replica,
					RollbackTo: rollbackTo,
					Template: api.PodTemplateSpec{
						Spec: api.PodSpec{
							SecurityContext: new(api.PodSecurityContext),
						},
					},
				},
			},
			deployment2: &appsv1.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{appsv1.DeprecatedRollbackTo: "2"},
				},
				Spec: appsv1.DeploymentSpec{
					Replicas: replica,
					Template: v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							SecurityContext: new(v1.PodSecurityContext),
						},
					},
				},
			},
		},
		"Deployment Conversion 2": {
			deployment1: &apps.Deployment{
				Spec: apps.DeploymentSpec{
					Replicas: *replica,
					Template: api.PodTemplateSpec{
						Spec: api.PodSpec{
							SecurityContext: new(api.PodSecurityContext),
						},
					},
				},
			},
			deployment2: &appsv1.Deployment{
				Spec: appsv1.DeploymentSpec{
					Replicas: replica,
					Template: v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							SecurityContext: new(v1.PodSecurityContext),
						},
					},
				},
			},
		},
	}

	for k, tc := range testcases {
		// apps -> v1beta2
		internal1 := &appsv1.Deployment{}
		if err := legacyscheme.Scheme.Convert(tc.deployment1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from apps to v1beta2", err)
		}
		if !apiequality.Semantic.DeepEqual(internal1, tc.deployment2) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "from apps to v1beta2", tc.deployment2, internal1)
		}

		// v1beta2 -> apps
		internal2 := &apps.Deployment{}
		if err := legacyscheme.Scheme.Convert(tc.deployment2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from v1beta2 to apps", err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.deployment1) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "from v1beta2 to apps", tc.deployment1, internal2)
		}
	}
}

func TestV1DeploymentSpecConversion(t *testing.T) {
	replica := utilpointer.Int32Ptr(2)
	revisionHistoryLimit := utilpointer.Int32Ptr(2)
	progressDeadlineSeconds := utilpointer.Int32Ptr(2)

	testcases := map[string]struct {
		deploymentSpec1 *apps.DeploymentSpec
		deploymentSpec2 *appsv1.DeploymentSpec
	}{
		"DeploymentSpec Conversion 1": {
			deploymentSpec1: &apps.DeploymentSpec{
				Replicas: *replica,
				Template: api.PodTemplateSpec{
					Spec: api.PodSpec{
						SecurityContext: new(api.PodSecurityContext),
					},
				},
			},
			deploymentSpec2: &appsv1.DeploymentSpec{
				Replicas: replica,
				Template: v1.PodTemplateSpec{
					Spec: v1.PodSpec{
						SecurityContext: new(v1.PodSecurityContext),
					},
				},
			},
		},
		"DeploymentSpec Conversion 2": {
			deploymentSpec1: &apps.DeploymentSpec{
				Replicas:             *replica,
				RevisionHistoryLimit: revisionHistoryLimit,
				MinReadySeconds:      2,
				Paused:               true,
				Template: api.PodTemplateSpec{
					Spec: api.PodSpec{
						SecurityContext: new(api.PodSecurityContext),
					},
				},
			},
			deploymentSpec2: &appsv1.DeploymentSpec{
				Replicas:             replica,
				RevisionHistoryLimit: revisionHistoryLimit,
				MinReadySeconds:      2,
				Paused:               true,
				Template: v1.PodTemplateSpec{
					Spec: v1.PodSpec{
						SecurityContext: new(v1.PodSecurityContext),
					},
				},
			},
		},
		"DeploymentSpec Conversion 3": {
			deploymentSpec1: &apps.DeploymentSpec{
				Replicas:                *replica,
				ProgressDeadlineSeconds: progressDeadlineSeconds,
				Template: api.PodTemplateSpec{
					Spec: api.PodSpec{
						SecurityContext: new(api.PodSecurityContext),
					},
				},
			},
			deploymentSpec2: &appsv1.DeploymentSpec{
				Replicas:                replica,
				ProgressDeadlineSeconds: progressDeadlineSeconds,
				Template: v1.PodTemplateSpec{
					Spec: v1.PodSpec{
						SecurityContext: new(v1.PodSecurityContext),
					},
				},
			},
		},
	}

	// apps -> appsv1
	for k, tc := range testcases {
		internal := &appsv1.DeploymentSpec{}
		if err := legacyscheme.Scheme.Convert(tc.deploymentSpec1, internal, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", "apps -> appsv1", k, err)
		}

		if !apiequality.Semantic.DeepEqual(internal, tc.deploymentSpec2) {
			t.Errorf("%q - %q: expected\n\t%+v, got \n\t%+v", "apps -> appsv1", k, tc.deploymentSpec2, internal)
		}
	}

	// appsv1 -> apps
	for k, tc := range testcases {
		internal := &apps.DeploymentSpec{}
		if err := legacyscheme.Scheme.Convert(tc.deploymentSpec2, internal, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", "appsv1 -> apps", k, err)
		}
		if !apiequality.Semantic.DeepEqual(internal, tc.deploymentSpec1) {
			t.Errorf("%q - %q: expected\n\t%+v, got \n\t%+v", "appsv1 -> apps", k, tc.deploymentSpec1, internal)
		}
	}

}

func TestV1DeploymentStrategyConversion(t *testing.T) {
	maxUnavailable := intstr.FromInt(2)
	maxSurge := intstr.FromInt(2)
	appsRollingUpdate := apps.RollingUpdateDeployment{MaxUnavailable: maxUnavailable, MaxSurge: maxSurge}
	appsv1RollingUpdate := appsv1.RollingUpdateDeployment{MaxUnavailable: &maxUnavailable, MaxSurge: &maxSurge}
	testcases := map[string]struct {
		deploymentStrategy1 *apps.DeploymentStrategy
		deploymentStrategy2 *appsv1.DeploymentStrategy
	}{
		"DeploymentStrategy Conversion 1": {
			deploymentStrategy1: &apps.DeploymentStrategy{Type: apps.DeploymentStrategyType("foo")},
			deploymentStrategy2: &appsv1.DeploymentStrategy{Type: appsv1.DeploymentStrategyType("foo")},
		},
		"DeploymentStrategy Conversion 2": {
			deploymentStrategy1: &apps.DeploymentStrategy{Type: apps.DeploymentStrategyType("foo"), RollingUpdate: &appsRollingUpdate},
			deploymentStrategy2: &appsv1.DeploymentStrategy{Type: appsv1.DeploymentStrategyType("foo"), RollingUpdate: &appsv1RollingUpdate},
		},
	}

	for k, tc := range testcases {
		// apps -> appsv1
		internal1 := &appsv1.DeploymentStrategy{}
		if err := legacyscheme.Scheme.Convert(tc.deploymentStrategy1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "apps -> appsv1", err)
		}
		if !apiequality.Semantic.DeepEqual(internal1, tc.deploymentStrategy2) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "apps -> appsv1", tc.deploymentStrategy2, internal1)
		}

		// appsv1 -> apps
		internal2 := &apps.DeploymentStrategy{}
		if err := legacyscheme.Scheme.Convert(tc.deploymentStrategy2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "appsv1 -> apps", err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.deploymentStrategy1) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "appsv1 -> apps", tc.deploymentStrategy1, internal2)
		}
	}
}

func TestV1RollingUpdateDeploymentConversion(t *testing.T) {
	nilIntStr := intstr.IntOrString{}
	maxUnavailable := intstr.FromInt(2)
	maxSurge := intstr.FromInt(2)
	testcases := map[string]struct {
		rollingUpdateDeployment1 *apps.RollingUpdateDeployment
		rollingUpdateDeployment2 *appsv1.RollingUpdateDeployment
	}{
		"RollingUpdateDeployment Conversion 1": {
			rollingUpdateDeployment1: &apps.RollingUpdateDeployment{},
			rollingUpdateDeployment2: &appsv1.RollingUpdateDeployment{MaxUnavailable: &nilIntStr, MaxSurge: &nilIntStr},
		},
		"RollingUpdateDeployment Conversion 2": {
			rollingUpdateDeployment1: &apps.RollingUpdateDeployment{MaxUnavailable: maxUnavailable},
			rollingUpdateDeployment2: &appsv1.RollingUpdateDeployment{MaxUnavailable: &maxUnavailable, MaxSurge: &nilIntStr},
		},
		"RollingUpdateDeployment Conversion 3": {
			rollingUpdateDeployment1: &apps.RollingUpdateDeployment{MaxSurge: maxSurge},
			rollingUpdateDeployment2: &appsv1.RollingUpdateDeployment{MaxSurge: &maxSurge, MaxUnavailable: &nilIntStr},
		},
		"RollingUpdateDeployment Conversion 4": {
			rollingUpdateDeployment1: &apps.RollingUpdateDeployment{MaxUnavailable: maxUnavailable, MaxSurge: maxSurge},
			rollingUpdateDeployment2: &appsv1.RollingUpdateDeployment{MaxUnavailable: &maxUnavailable, MaxSurge: &maxSurge},
		},
	}

	for k, tc := range testcases {
		// apps -> appsv1
		internal1 := &appsv1.RollingUpdateDeployment{}
		if err := legacyscheme.Scheme.Convert(tc.rollingUpdateDeployment1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "apps -> appsv1", err)
		}
		if !apiequality.Semantic.DeepEqual(internal1, tc.rollingUpdateDeployment2) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "apps -> appsv1", tc.rollingUpdateDeployment2, internal1)
		}

		// appsv1 -> apps
		internal2 := &apps.RollingUpdateDeployment{}
		if err := legacyscheme.Scheme.Convert(tc.rollingUpdateDeployment2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "appsv1 -> apps", err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.rollingUpdateDeployment1) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "appsv1 -> apps", tc.rollingUpdateDeployment1, internal2)
		}
	}
}

func TestV1ReplicaSetSpecConversion(t *testing.T) {
	replicas := new(int32)
	*replicas = 2
	matchExpressions := []metav1.LabelSelectorRequirement{
		{Key: "foo", Operator: metav1.LabelSelectorOpIn, Values: []string{"foo"}},
	}
	matchLabels := map[string]string{"foo": "bar"}
	selector := &metav1.LabelSelector{MatchLabels: matchLabels, MatchExpressions: matchExpressions}

	testcases := map[string]struct {
		replicaset1 *apps.ReplicaSetSpec
		replicaset2 *appsv1.ReplicaSetSpec
	}{
		"ReplicaSetSpec Conversion 1": {
			replicaset1: &apps.ReplicaSetSpec{
				Replicas:        *replicas,
				MinReadySeconds: 2,
				Template: api.PodTemplateSpec{
					Spec: api.PodSpec{
						SecurityContext: new(api.PodSecurityContext),
					},
				},
			},
			replicaset2: &appsv1.ReplicaSetSpec{
				Replicas:        replicas,
				MinReadySeconds: 2,
				Template: v1.PodTemplateSpec{
					Spec: v1.PodSpec{
						SecurityContext: new(v1.PodSecurityContext),
					},
				},
			},
		},
		"ReplicaSetSpec Conversion 2": {
			replicaset1: &apps.ReplicaSetSpec{
				Replicas: *replicas,
				Selector: selector,
				Template: api.PodTemplateSpec{
					Spec: api.PodSpec{
						SecurityContext: new(api.PodSecurityContext),
					},
				},
			},
			replicaset2: &appsv1.ReplicaSetSpec{
				Replicas: replicas,
				Selector: selector,
				Template: v1.PodTemplateSpec{
					Spec: v1.PodSpec{
						SecurityContext: new(v1.PodSecurityContext),
					},
				},
			},
		},
	}

	for k, tc := range testcases {
		// apps -> appsv1
		internal1 := &appsv1.ReplicaSetSpec{}
		if err := legacyscheme.Scheme.Convert(tc.replicaset1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "apps -> appsv1", err)
		}

		if !apiequality.Semantic.DeepEqual(internal1, tc.replicaset2) {
			t.Errorf("%q - %q: expected\n\t%+v, got \n\t%+v", k, "apps -> appsv1", tc.replicaset2, internal1)
		}

		// appsv1 -> apps
		internal2 := &apps.ReplicaSetSpec{}
		if err := legacyscheme.Scheme.Convert(tc.replicaset2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "appsv1 -> apps", err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.replicaset1) {
			t.Errorf("%q - %q: expected\n\t%+v, got \n\t%+v", k, "appsv1 -> apps", tc.replicaset1, internal2)
		}
	}
}
