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
	"k8s.io/kubernetes/pkg/apis/extensions"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
)

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

func TestV1DeploymentConversion(t *testing.T) {
	replica := newInt32(2)
	rollbackTo := new(extensions.RollbackConfig)
	rollbackTo.Revision = int64(2)
	testcases := map[string]struct {
		deployment1 *extensions.Deployment
		deployment2 *appsv1.Deployment
	}{
		"Deployment Conversion 1": {
			deployment1: &extensions.Deployment{
				Spec: extensions.DeploymentSpec{
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
			deployment1: &extensions.Deployment{
				Spec: extensions.DeploymentSpec{
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
		// extensions -> v1beta2
		internal1 := &appsv1.Deployment{}
		if err := api.Scheme.Convert(tc.deployment1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from extensions to v1beta2", err)
		}
		if !apiequality.Semantic.DeepEqual(internal1, tc.deployment2) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "from extensions to v1beta2", tc.deployment2, internal1)
		}

		// v1beta2 -> extensions
		internal2 := &extensions.Deployment{}
		if err := api.Scheme.Convert(tc.deployment2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from v1beta2 to extensions", err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.deployment1) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "from v1beta2 to extensions", tc.deployment1, internal2)
		}
	}
}

func TestV1DeploymentSpecConversion(t *testing.T) {
	replica := newInt32(2)
	revisionHistoryLimit := newInt32(2)
	progressDeadlineSeconds := newInt32(2)

	testcases := map[string]struct {
		deploymentSpec1 *extensions.DeploymentSpec
		deploymentSpec2 *appsv1.DeploymentSpec
	}{
		"DeploymentSpec Conversion 1": {
			deploymentSpec1: &extensions.DeploymentSpec{
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
			deploymentSpec1: &extensions.DeploymentSpec{
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
			deploymentSpec1: &extensions.DeploymentSpec{
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

	// extensions -> appsv1
	for k, tc := range testcases {
		internal := &appsv1.DeploymentSpec{}
		if err := api.Scheme.Convert(tc.deploymentSpec1, internal, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", "extensions -> appsv1", k, err)
		}

		if !apiequality.Semantic.DeepEqual(internal, tc.deploymentSpec2) {
			t.Errorf("%q - %q: expected\n\t%+v, got \n\t%+v", "extensions -> appsv1", k, tc.deploymentSpec2, internal)
		}
	}

	// appsv1 -> extensions
	for k, tc := range testcases {
		internal := &extensions.DeploymentSpec{}
		if err := api.Scheme.Convert(tc.deploymentSpec2, internal, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", "appsv1 -> extensions", k, err)
		}
		if !apiequality.Semantic.DeepEqual(internal, tc.deploymentSpec1) {
			t.Errorf("%q - %q: expected\n\t%+v, got \n\t%+v", "appsv1 -> extensions", k, tc.deploymentSpec1, internal)
		}
	}

}

func TestV1DeploymentStrategyConversion(t *testing.T) {
	maxUnavailable := intstr.FromInt(2)
	maxSurge := intstr.FromInt(2)
	extensionsRollingUpdate := extensions.RollingUpdateDeployment{MaxUnavailable: maxUnavailable, MaxSurge: maxSurge}
	appsv1RollingUpdate := appsv1.RollingUpdateDeployment{MaxUnavailable: &maxUnavailable, MaxSurge: &maxSurge}
	testcases := map[string]struct {
		deploymentStrategy1 *extensions.DeploymentStrategy
		deploymentStrategy2 *appsv1.DeploymentStrategy
	}{
		"DeploymentStrategy Conversion 1": {
			deploymentStrategy1: &extensions.DeploymentStrategy{Type: extensions.DeploymentStrategyType("foo")},
			deploymentStrategy2: &appsv1.DeploymentStrategy{Type: appsv1.DeploymentStrategyType("foo")},
		},
		"DeploymentStrategy Conversion 2": {
			deploymentStrategy1: &extensions.DeploymentStrategy{Type: extensions.DeploymentStrategyType("foo"), RollingUpdate: &extensionsRollingUpdate},
			deploymentStrategy2: &appsv1.DeploymentStrategy{Type: appsv1.DeploymentStrategyType("foo"), RollingUpdate: &appsv1RollingUpdate},
		},
	}

	for k, tc := range testcases {
		// extensions -> appsv1
		internal1 := &appsv1.DeploymentStrategy{}
		if err := api.Scheme.Convert(tc.deploymentStrategy1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "extensions -> appsv1", err)
		}
		if !apiequality.Semantic.DeepEqual(internal1, tc.deploymentStrategy2) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "extensions -> appsv1", tc.deploymentStrategy2, internal1)
		}

		// appsv1 -> extensions
		internal2 := &extensions.DeploymentStrategy{}
		if err := api.Scheme.Convert(tc.deploymentStrategy2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "appsv1 -> extensions", err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.deploymentStrategy1) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "appsv1 -> extensions", tc.deploymentStrategy1, internal2)
		}
	}
}

func TestV1RollingUpdateDeploymentConversion(t *testing.T) {
	nilIntStr := intstr.IntOrString{}
	maxUnavailable := intstr.FromInt(2)
	maxSurge := intstr.FromInt(2)
	testcases := map[string]struct {
		rollingUpdateDeployment1 *extensions.RollingUpdateDeployment
		rollingUpdateDeployment2 *appsv1.RollingUpdateDeployment
	}{
		"RollingUpdateDeployment Conversion 1": {
			rollingUpdateDeployment1: &extensions.RollingUpdateDeployment{},
			rollingUpdateDeployment2: &appsv1.RollingUpdateDeployment{MaxUnavailable: &nilIntStr, MaxSurge: &nilIntStr},
		},
		"RollingUpdateDeployment Conversion 2": {
			rollingUpdateDeployment1: &extensions.RollingUpdateDeployment{MaxUnavailable: maxUnavailable},
			rollingUpdateDeployment2: &appsv1.RollingUpdateDeployment{MaxUnavailable: &maxUnavailable, MaxSurge: &nilIntStr},
		},
		"RollingUpdateDeployment Conversion 3": {
			rollingUpdateDeployment1: &extensions.RollingUpdateDeployment{MaxSurge: maxSurge},
			rollingUpdateDeployment2: &appsv1.RollingUpdateDeployment{MaxSurge: &maxSurge, MaxUnavailable: &nilIntStr},
		},
		"RollingUpdateDeployment Conversion 4": {
			rollingUpdateDeployment1: &extensions.RollingUpdateDeployment{MaxUnavailable: maxUnavailable, MaxSurge: maxSurge},
			rollingUpdateDeployment2: &appsv1.RollingUpdateDeployment{MaxUnavailable: &maxUnavailable, MaxSurge: &maxSurge},
		},
	}

	for k, tc := range testcases {
		// extensions -> appsv1
		internal1 := &appsv1.RollingUpdateDeployment{}
		if err := api.Scheme.Convert(tc.rollingUpdateDeployment1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "extensions -> appsv1", err)
		}
		if !apiequality.Semantic.DeepEqual(internal1, tc.rollingUpdateDeployment2) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "extensions -> appsv1", tc.rollingUpdateDeployment2, internal1)
		}

		// appsv1 -> extensions
		internal2 := &extensions.RollingUpdateDeployment{}
		if err := api.Scheme.Convert(tc.rollingUpdateDeployment2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "appsv1 -> extensions", err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.rollingUpdateDeployment1) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "appsv1 -> extensions", tc.rollingUpdateDeployment1, internal2)
		}
	}
}
