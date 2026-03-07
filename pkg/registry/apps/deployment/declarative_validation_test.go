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
	"k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/apps"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestDeclarativeValidateForDeclarative(t *testing.T) {
	ctx := request.WithRequestInfo(request.NewContext(), &request.RequestInfo{
		APIGroup:   "apps",
		APIVersion: "v1",
		Resource:   "deployments",
	})
	testCases := map[string]struct {
		input        apps.Deployment
		expectedErrs field.ErrorList
	}{
		"valid RollingUpdate strategy": {
			input: mkDeployment(apps.RollingUpdateDeploymentStrategyType, &apps.RollingUpdateDeployment{
				MaxUnavailable: intstr.FromInt32(1),
				MaxSurge:       intstr.FromInt32(1),
			}),
		},
		"valid Recreate strategy": {
			input: mkDeployment(apps.RecreateDeploymentStrategyType, nil),
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestDeclarativeValidateUpdateForDeclarative(t *testing.T) {
	ctx := request.WithRequestInfo(request.NewContext(), &request.RequestInfo{
		APIGroup:   "apps",
		APIVersion: "v1",
		Resource:   "deployments",
	})
	testCases := map[string]struct {
		oldObj       apps.Deployment
		updateObj    apps.Deployment
		expectedErrs field.ErrorList
	}{
		"valid update RollingUpdate to Recreate": {
			oldObj: mkDeployment(apps.RollingUpdateDeploymentStrategyType, &apps.RollingUpdateDeployment{
				MaxUnavailable: intstr.FromInt32(1),
				MaxSurge:       intstr.FromInt32(1),
			}),
			updateObj: mkDeployment(apps.RecreateDeploymentStrategyType, nil),
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.oldObj.ResourceVersion = "1"
			tc.updateObj.ResourceVersion = "1"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.oldObj, &tc.updateObj, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func mkDeployment(strategyType apps.DeploymentStrategyType, rollingUpdate *apps.RollingUpdateDeployment) apps.Deployment {
	return apps.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-deployment",
			Namespace: "default",
		},
		Spec: apps.DeploymentSpec{
			Replicas: 1,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": "test"},
			},
			Template: api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"app": "test"},
				},
				Spec: api.PodSpec{
					RestartPolicy:                 api.RestartPolicyAlways,
					TerminationGracePeriodSeconds: func(i int64) *int64 { return &i }(30),
					DNSPolicy:                     api.DNSClusterFirst,
					Containers: []api.Container{
						{Name: "c", Image: "img", TerminationMessagePolicy: api.TerminationMessageFallbackToLogsOnError, ImagePullPolicy: api.PullIfNotPresent},
					},
				},
			},
			Strategy: apps.DeploymentStrategy{
				Type:          strategyType,
				RollingUpdate: rollingUpdate,
			},
		},
	}
}
