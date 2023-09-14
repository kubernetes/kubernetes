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

package v1beta2_test

import (
	"testing"

	"k8s.io/api/apps/v1beta2"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	api "k8s.io/kubernetes/pkg/apis/core"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilpointer "k8s.io/utils/pointer"
)

func TestV1beta2StatefulSetSpecConversion(t *testing.T) {
	replicas := utilpointer.Int32(2)
	selector := &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}}
	v1beta2Template := v1.PodTemplateSpec{
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
		stsSepc2 *v1beta2.StatefulSetSpec
	}{
		"StatefulSetSpec Conversion 1": {
			stsSpec1: &apps.StatefulSetSpec{
				Replicas: *replicas,
				Template: apiTemplate,
			},
			stsSepc2: &v1beta2.StatefulSetSpec{
				Replicas: replicas,
				Template: v1beta2Template,
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
			stsSepc2: &v1beta2.StatefulSetSpec{
				Replicas:            replicas,
				Selector:            selector,
				Template:            v1beta2Template,
				ServiceName:         "foo",
				PodManagementPolicy: v1beta2.PodManagementPolicyType("bar"),
			},
		},
	}

	for k, tc := range testcases {
		// apps -> v1beta2
		internal1 := &v1beta2.StatefulSetSpec{}
		if err := legacyscheme.Scheme.Convert(tc.stsSpec1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from extensions to v1beta2", err)
		}

		if !apiequality.Semantic.DeepEqual(internal1, tc.stsSepc2) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "from extensions to v1beta2", tc.stsSepc2, internal1)
		}

		// v1beta2 -> apps
		internal2 := &apps.StatefulSetSpec{}
		if err := legacyscheme.Scheme.Convert(tc.stsSepc2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from v1beta2 to extensions", err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.stsSpec1) {
			t.Errorf("%q- %q: expected\n\t%#v, got \n\t%#v", k, "from v1beta2 to extensions", tc.stsSpec1, internal2)
		}
	}
}

func TestV1beta2StatefulSetUpdateStrategyConversion(t *testing.T) {
	partition := utilpointer.Int32(2)
	v1beta2rollingUpdate := new(v1beta2.RollingUpdateStatefulSetStrategy)
	v1beta2rollingUpdate.Partition = partition
	appsrollingUpdate := new(apps.RollingUpdateStatefulSetStrategy)
	appsrollingUpdate.Partition = *partition
	testcases := map[string]struct {
		stsUpdateStrategy1 *apps.StatefulSetUpdateStrategy
		stsUpdateStrategy2 *v1beta2.StatefulSetUpdateStrategy
	}{
		"StatefulSetUpdateStrategy Conversion 1": {
			stsUpdateStrategy1: &apps.StatefulSetUpdateStrategy{Type: apps.StatefulSetUpdateStrategyType("foo")},
			stsUpdateStrategy2: &v1beta2.StatefulSetUpdateStrategy{Type: v1beta2.StatefulSetUpdateStrategyType("foo")},
		},
		"StatefulSetUpdateStrategy Conversion 2": {
			stsUpdateStrategy1: &apps.StatefulSetUpdateStrategy{
				Type:          apps.StatefulSetUpdateStrategyType("foo"),
				RollingUpdate: appsrollingUpdate,
			},
			stsUpdateStrategy2: &v1beta2.StatefulSetUpdateStrategy{
				Type:          v1beta2.StatefulSetUpdateStrategyType("foo"),
				RollingUpdate: v1beta2rollingUpdate,
			},
		},
	}

	for k, tc := range testcases {
		// apps -> v1beta2
		internal1 := &v1beta2.StatefulSetUpdateStrategy{}
		if err := legacyscheme.Scheme.Convert(tc.stsUpdateStrategy1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", "apps -> v1beta2", k, err)
		}

		if !apiequality.Semantic.DeepEqual(internal1, tc.stsUpdateStrategy2) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", "apps -> v1beta2", k, tc.stsUpdateStrategy2, internal1)
		}

		// v1beta2 -> apps
		internal2 := &apps.StatefulSetUpdateStrategy{}
		if err := legacyscheme.Scheme.Convert(tc.stsUpdateStrategy2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", "v1beta2 -> apps", k, err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.stsUpdateStrategy1) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", "v1beta2 -> apps", k, tc.stsUpdateStrategy1, internal2)
		}
	}
}

func TestV1beta2RollingUpdateDaemonSetConversion(t *testing.T) {
	intorstr := intstr.FromInt32(1)
	maxSurge := intstr.FromInt32(0)
	testcases := map[string]struct {
		rollingUpdateDs1 *apps.RollingUpdateDaemonSet
		rollingUpdateDs2 *v1beta2.RollingUpdateDaemonSet
	}{
		"RollingUpdateDaemonSet Conversion 2": {
			rollingUpdateDs1: &apps.RollingUpdateDaemonSet{MaxUnavailable: intorstr, MaxSurge: maxSurge},
			rollingUpdateDs2: &v1beta2.RollingUpdateDaemonSet{MaxUnavailable: &intorstr, MaxSurge: &maxSurge},
		},
	}

	for k, tc := range testcases {
		// extensions -> v1beta2
		internal1 := &v1beta2.RollingUpdateDaemonSet{}
		if err := legacyscheme.Scheme.Convert(tc.rollingUpdateDs1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from extensions to v1beta2", err)
		}
		if !apiequality.Semantic.DeepEqual(internal1, tc.rollingUpdateDs2) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "from extensions to v1beta2", tc.rollingUpdateDs2, internal1)
		}

		// v1beta2 -> extensions
		internal2 := &apps.RollingUpdateDaemonSet{}
		if err := legacyscheme.Scheme.Convert(tc.rollingUpdateDs2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from v1beta2 to extensions", err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.rollingUpdateDs1) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "from v1beta2 to extensions", tc.rollingUpdateDs1, internal2)
		}
	}
}

func TestV1beta2StatefulSetStatusConversion(t *testing.T) {
	observedGeneration := new(int64)
	*observedGeneration = 2
	collisionCount := new(int32)
	*collisionCount = 1
	testcases := map[string]struct {
		stsStatus1 *apps.StatefulSetStatus
		stsStatus2 *v1beta2.StatefulSetStatus
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
			stsStatus2: &v1beta2.StatefulSetStatus{
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
			stsStatus2: &v1beta2.StatefulSetStatus{
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
		// apps -> v1beta2
		internal1 := &v1beta2.StatefulSetStatus{}
		if err := legacyscheme.Scheme.Convert(tc.stsStatus1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from apps to v1beta2", err)
		}

		if !apiequality.Semantic.DeepEqual(internal1, tc.stsStatus2) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "from apps to v1beta2", tc.stsStatus2, internal1)
		}

		// v1beta2 -> apps
		internal2 := &apps.StatefulSetStatus{}
		if err := legacyscheme.Scheme.Convert(tc.stsStatus2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from v1beta2 to apps", err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.stsStatus1) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "from v1beta2 to apps", tc.stsStatus1, internal2)
		}
	}
}

func TestV1beta2DeploymentConversion(t *testing.T) {
	replica := utilpointer.Int32(2)
	rollbackTo := new(apps.RollbackConfig)
	rollbackTo.Revision = int64(2)
	testcases := map[string]struct {
		deployment1 *apps.Deployment
		deployment2 *v1beta2.Deployment
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
			deployment2: &v1beta2.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{v1beta2.DeprecatedRollbackTo: "2"},
				},
				Spec: v1beta2.DeploymentSpec{
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
			deployment2: &v1beta2.Deployment{
				Spec: v1beta2.DeploymentSpec{
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
		// extensions -> v1beta2
		internal1 := &v1beta2.Deployment{}
		if err := legacyscheme.Scheme.Convert(tc.deployment1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from extensions to v1beta2", err)
		}
		if !apiequality.Semantic.DeepEqual(internal1, tc.deployment2) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "from extensions to v1beta2", tc.deployment2, internal1)
		}

		// v1beta2 -> extensions
		internal2 := &apps.Deployment{}
		if err := legacyscheme.Scheme.Convert(tc.deployment2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from v1beta2 to extensions", err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.deployment1) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "from v1beta2 to extensions", tc.deployment1, internal2)
		}
	}
}

func TestV1beta2ScaleStatusConversion(t *testing.T) {
	matchLabels := map[string]string{"foo": "bar"}
	selector1 := &metav1.LabelSelector{MatchLabels: matchLabels}
	labelsSelector1, _ := metav1.LabelSelectorAsSelector(selector1)

	matchExpressions := []metav1.LabelSelectorRequirement{
		{Key: "foo", Operator: metav1.LabelSelectorOpIn, Values: []string{"foo"}},
	}
	selector2 := &metav1.LabelSelector{MatchLabels: matchLabels, MatchExpressions: matchExpressions}
	labelsSelector2, _ := metav1.LabelSelectorAsSelector(selector2)

	testcases := map[string]struct {
		scaleStatus1 *autoscaling.ScaleStatus
		scaleStatus2 *v1beta2.ScaleStatus
	}{
		"ScaleStatus Conversion 1": {
			scaleStatus1: &autoscaling.ScaleStatus{Replicas: 2},
			scaleStatus2: &v1beta2.ScaleStatus{Replicas: 2},
		},
		"ScaleStatus Conversion 2": {
			scaleStatus1: &autoscaling.ScaleStatus{Replicas: 2, Selector: labelsSelector1.String()},
			scaleStatus2: &v1beta2.ScaleStatus{Replicas: 2, Selector: matchLabels, TargetSelector: labelsSelector1.String()},
		},
		"ScaleStatus Conversion 3": {
			scaleStatus1: &autoscaling.ScaleStatus{Replicas: 2, Selector: labelsSelector2.String()},
			scaleStatus2: &v1beta2.ScaleStatus{Replicas: 2, Selector: map[string]string{}, TargetSelector: labelsSelector2.String()},
		},
	}

	for k, tc := range testcases {
		// autoscaling -> v1beta2
		internal1 := &v1beta2.ScaleStatus{}
		if err := legacyscheme.Scheme.Convert(tc.scaleStatus1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "autoscaling -> v1beta2", err)
		}

		if !apiequality.Semantic.DeepEqual(internal1, tc.scaleStatus2) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "autoscaling -> v1beta2", tc.scaleStatus2, internal1)
		}

		// v1beta2 -> autoscaling
		internal2 := &autoscaling.ScaleStatus{}
		if err := legacyscheme.Scheme.Convert(tc.scaleStatus2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "v1beta2 -> autoscaling", err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.scaleStatus1) {
			t.Errorf("%q - %q: expected\n\t%+v, got \n\t%+v", k, "v1beta2 -> autoscaling", tc.scaleStatus1, internal2)
		}
	}
}

func TestV1beta2DeploymentSpecConversion(t *testing.T) {
	replica := utilpointer.Int32(2)
	revisionHistoryLimit := utilpointer.Int32(2)
	progressDeadlineSeconds := utilpointer.Int32(2)

	testcases := map[string]struct {
		deploymentSpec1 *apps.DeploymentSpec
		deploymentSpec2 *v1beta2.DeploymentSpec
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
			deploymentSpec2: &v1beta2.DeploymentSpec{
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
			deploymentSpec2: &v1beta2.DeploymentSpec{
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
			deploymentSpec2: &v1beta2.DeploymentSpec{
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

	// extensions -> v1beta2
	for k, tc := range testcases {
		internal := &v1beta2.DeploymentSpec{}
		if err := legacyscheme.Scheme.Convert(tc.deploymentSpec1, internal, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", "extensions -> v1beta2", k, err)
		}

		if !apiequality.Semantic.DeepEqual(internal, tc.deploymentSpec2) {
			t.Errorf("%q - %q: expected\n\t%+v, got \n\t%+v", "extensions -> v1beta2", k, tc.deploymentSpec2, internal)
		}
	}

	// v1beta2 -> extensions
	for k, tc := range testcases {
		internal := &apps.DeploymentSpec{}
		if err := legacyscheme.Scheme.Convert(tc.deploymentSpec2, internal, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", "v1beta2 -> extensions", k, err)
		}
		if !apiequality.Semantic.DeepEqual(internal, tc.deploymentSpec1) {
			t.Errorf("%q - %q: expected\n\t%+v, got \n\t%+v", "v1beta2 -> extensions", k, tc.deploymentSpec1, internal)
		}
	}

}

func TestV1beta2DeploymentStrategyConversion(t *testing.T) {
	maxUnavailable := intstr.FromInt32(2)
	maxSurge := intstr.FromInt32(2)
	extensionsRollingUpdate := apps.RollingUpdateDeployment{MaxUnavailable: maxUnavailable, MaxSurge: maxSurge}
	v1beta2RollingUpdate := v1beta2.RollingUpdateDeployment{MaxUnavailable: &maxUnavailable, MaxSurge: &maxSurge}
	testcases := map[string]struct {
		deploymentStrategy1 *apps.DeploymentStrategy
		deploymentStrategy2 *v1beta2.DeploymentStrategy
	}{
		"DeploymentStrategy Conversion 1": {
			deploymentStrategy1: &apps.DeploymentStrategy{Type: apps.DeploymentStrategyType("foo")},
			deploymentStrategy2: &v1beta2.DeploymentStrategy{Type: v1beta2.DeploymentStrategyType("foo")},
		},
		"DeploymentStrategy Conversion 2": {
			deploymentStrategy1: &apps.DeploymentStrategy{Type: apps.DeploymentStrategyType("foo"), RollingUpdate: &extensionsRollingUpdate},
			deploymentStrategy2: &v1beta2.DeploymentStrategy{Type: v1beta2.DeploymentStrategyType("foo"), RollingUpdate: &v1beta2RollingUpdate},
		},
	}

	for k, tc := range testcases {
		// extensions -> v1beta2
		internal1 := &v1beta2.DeploymentStrategy{}
		if err := legacyscheme.Scheme.Convert(tc.deploymentStrategy1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "extensions -> v1beta2", err)
		}
		if !apiequality.Semantic.DeepEqual(internal1, tc.deploymentStrategy2) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "extensions -> v1beta2", tc.deploymentStrategy2, internal1)
		}

		// v1beta2 -> extensions
		internal2 := &apps.DeploymentStrategy{}
		if err := legacyscheme.Scheme.Convert(tc.deploymentStrategy2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "v1beta2 -> extensions", err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.deploymentStrategy1) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "v1beta2 -> extensions", tc.deploymentStrategy1, internal2)
		}
	}
}

func TestV1beta2RollingUpdateDeploymentConversion(t *testing.T) {
	nilIntStr := intstr.IntOrString{}
	maxUnavailable := intstr.FromInt32(2)
	maxSurge := intstr.FromInt32(2)
	testcases := map[string]struct {
		rollingUpdateDeployment1 *apps.RollingUpdateDeployment
		rollingUpdateDeployment2 *v1beta2.RollingUpdateDeployment
	}{
		"RollingUpdateDeployment Conversion 1": {
			rollingUpdateDeployment1: &apps.RollingUpdateDeployment{},
			rollingUpdateDeployment2: &v1beta2.RollingUpdateDeployment{MaxUnavailable: &nilIntStr, MaxSurge: &nilIntStr},
		},
		"RollingUpdateDeployment Conversion 2": {
			rollingUpdateDeployment1: &apps.RollingUpdateDeployment{MaxUnavailable: maxUnavailable},
			rollingUpdateDeployment2: &v1beta2.RollingUpdateDeployment{MaxUnavailable: &maxUnavailable, MaxSurge: &nilIntStr},
		},
		"RollingUpdateDeployment Conversion 3": {
			rollingUpdateDeployment1: &apps.RollingUpdateDeployment{MaxSurge: maxSurge},
			rollingUpdateDeployment2: &v1beta2.RollingUpdateDeployment{MaxSurge: &maxSurge, MaxUnavailable: &nilIntStr},
		},
		"RollingUpdateDeployment Conversion 4": {
			rollingUpdateDeployment1: &apps.RollingUpdateDeployment{MaxUnavailable: maxUnavailable, MaxSurge: maxSurge},
			rollingUpdateDeployment2: &v1beta2.RollingUpdateDeployment{MaxUnavailable: &maxUnavailable, MaxSurge: &maxSurge},
		},
	}

	for k, tc := range testcases {
		// extensions -> v1beta2
		internal1 := &v1beta2.RollingUpdateDeployment{}
		if err := legacyscheme.Scheme.Convert(tc.rollingUpdateDeployment1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "extensions -> v1beta2", err)
		}
		if !apiequality.Semantic.DeepEqual(internal1, tc.rollingUpdateDeployment2) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "extensions -> v1beta2", tc.rollingUpdateDeployment2, internal1)
		}

		// v1beta2 -> extensions
		internal2 := &apps.RollingUpdateDeployment{}
		if err := legacyscheme.Scheme.Convert(tc.rollingUpdateDeployment2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "v1beta2 -> extensions", err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.rollingUpdateDeployment1) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "v1beta2 -> extensions", tc.rollingUpdateDeployment1, internal2)
		}
	}
}

func TestV1beta2ReplicaSetSpecConversion(t *testing.T) {
	replicas := new(int32)
	*replicas = 2
	matchExpressions := []metav1.LabelSelectorRequirement{
		{Key: "foo", Operator: metav1.LabelSelectorOpIn, Values: []string{"foo"}},
	}
	matchLabels := map[string]string{"foo": "bar"}
	selector := &metav1.LabelSelector{MatchLabels: matchLabels, MatchExpressions: matchExpressions}

	testcases := map[string]struct {
		replicaset1 *apps.ReplicaSetSpec
		replicaset2 *v1beta2.ReplicaSetSpec
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
			replicaset2: &v1beta2.ReplicaSetSpec{
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
			replicaset2: &v1beta2.ReplicaSetSpec{
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
		// extensions -> v1beta2
		internal1 := &v1beta2.ReplicaSetSpec{}
		if err := legacyscheme.Scheme.Convert(tc.replicaset1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "extensions -> v1beta2", err)
		}

		if !apiequality.Semantic.DeepEqual(internal1, tc.replicaset2) {
			t.Errorf("%q - %q: expected\n\t%+v, got \n\t%+v", k, "extensions -> v1beta2", tc.replicaset2, internal1)
		}

		// v1beta2 -> extensions
		internal2 := &apps.ReplicaSetSpec{}
		if err := legacyscheme.Scheme.Convert(tc.replicaset2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "v1beta2 -> extensions", err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.replicaset1) {
			t.Errorf("%q - %q: expected\n\t%+v, got \n\t%+v", k, "v1beta2 -> extensions", tc.replicaset1, internal2)
		}
	}
}
