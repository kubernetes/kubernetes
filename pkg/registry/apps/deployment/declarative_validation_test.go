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

package deployment

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/core"
)

var apiVersions = []string{"v1", "v1beta1", "v1beta2"}

func TestValidateUpdateForDeclarative(t *testing.T) {
	for _, v := range apiVersions {
		t.Run("version="+v, func(t *testing.T) {
			testValidateUpdateForDeclarative(t, v)
		})
	}
}

func testValidateUpdateForDeclarative(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(
		genericapirequest.NewDefaultContext(),
		&genericapirequest.RequestInfo{
			APIGroup:   "apps",
			APIVersion: apiVersion,
			Resource:   "deployments",
		},
	)

	testCases := map[string]struct {
		old, update  apps.Deployment
		expectedErrs field.ErrorList
	}{
		"valid update": {
			old:    mkValidDeployment(),
			update: mkValidDeployment(),
		},
		"valid update when deployment spec is changed": {
			old:    mkValidDeployment(),
			update: mkValidDeployment(tweakDeploymentSpec()),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.selector"), "", "").WithOrigin("immutable"),
			},
		},
		// TODO: Add more test cases
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			tc.old.ResourceVersion = "1"
			tc.update.ResourceVersion = "2"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func tweakDeploymentSpec() func(*apps.Deployment) {
	return func(d *apps.Deployment) {
		d.Spec.Selector = &metav1.LabelSelector{
			MatchLabels: map[string]string{
				"app": "modified",
			},
		}
	}
}

func mkValidDeployment(tweaks ...func(*apps.Deployment)) apps.Deployment {
	r := apps.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-revision",
			Namespace: "default",
		},
		Spec: apps.DeploymentSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"app": "test",
				},
			},
			Strategy: apps.DeploymentStrategy{
				Type: apps.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &apps.RollingUpdateDeployment{
					MaxSurge:       intstr.FromInt32(1),
					MaxUnavailable: intstr.FromInt32(1),
				},
			},
			Template: core.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"app": "test",
					},
				},
				Spec: core.PodSpec{
					RestartPolicy:                 core.RestartPolicyAlways,
					DNSPolicy:                     core.DNSClusterFirst,
					TerminationGracePeriodSeconds: new(int64),
					Containers: []core.Container{
						{
							Name:                     "test-container",
							Image:                    "test-image",
							ImagePullPolicy:          core.PullIfNotPresent,
							TerminationMessagePolicy: core.TerminationMessageReadFile,
						},
					},
				},
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&r)
	}
	return r
}
