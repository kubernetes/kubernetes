/*
Copyright The Kubernetes Authors.

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
	"context"
	"testing"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/test/coverage"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	validationmetrics "k8s.io/apiserver/pkg/validation"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/apps"
	api "k8s.io/kubernetes/pkg/apis/core"
	registry "k8s.io/kubernetes/pkg/registry/apps/deployment"
	"k8s.io/utils/ptr"
)

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "apps",
				APIVersion:        apiVersion,
				Resource:          "deployments",
				IsResourceRequest: true,
				Verb:              "create",
			})

			testCases := map[string]struct {
				input        apps.Deployment
				expectedErrs field.ErrorList
				skipVersions []string
				allRulesOnly bool
			}{
				"valid": {
					input: mkValidDeployment(),
				},
				"valid toleration key": {
					input: mkValidDeployment(tweakTolerations(api.Toleration{Key: "example.com/valid-key", Operator: api.TolerationOpExists})),
				},
				"valid toleration key without prefix": {
					input: mkValidDeployment(tweakTolerations(api.Toleration{Key: "simple-key", Operator: api.TolerationOpExists})),
				},
				"invalid toleration key format": {
					input: mkValidDeployment(tweakTolerations(api.Toleration{Key: "invalid key", Operator: api.TolerationOpExists})),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "template", "spec", "tolerations").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key").MarkAlpha(),
					},
				},
				"selector required": {
					input: mkValidDeployment(tweakSelector(nil)),
					expectedErrs: field.ErrorList{
						field.Required(field.NewPath("spec").Child("selector"), "").MarkCoveredByDeclarative().MarkAlpha(),
						field.Invalid(field.NewPath("spec").Child("template").Child("metadata").Child("labels"), nil, "").MarkFromImperative(),
					},
					allRulesOnly: true,
				},
			}
			for name, tc := range testCases {
				t.Run(name, func(t *testing.T) {
					for _, version := range tc.skipVersions {
						if apiVersion == version {
							t.Skipf("not applicable to %s", apiVersion)
						}
					}
					if tc.allRulesOnly {
						verifyDeclarativeValidateAllRules(t, ctx, apiVersion, &tc.input, tc.expectedErrs)
						return
					}
					apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, registry.Strategy, tc.expectedErrs)
				})
			}
		})
	}
}

func verifyDeclarativeValidateAllRules(t *testing.T, ctx context.Context, apiVersion string, input *apps.Deployment, expectedErrs field.ErrorList) {
	t.Helper()

	testCtx := rest.WithAllDeclarativeEnforcedForTest(ctx)
	legacyregistry.Reset()
	defer legacyregistry.Reset()
	validationmetrics.ResetValidationMetricsInstance()
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.DeclarativeValidation:     true,
		features.DeclarativeValidationBeta: true,
	})

	errs := registry.Strategy.Validate(testCtx, input)
	errs = registry.Strategy.ValidateDeclaratively(
		testCtx,
		input,
		nil,
		errs,
		operation.Create,
		registry.Strategy.DeclarativeValidationConfig(testCtx, input, nil),
	)

	field.ErrorMatcher{}.ByType().ByOrigin().ByField().ByValidationStabilityLevel().BySource().Test(t, expectedErrs, errs)
	coverage.RecordObservedRules(schema.GroupVersionKind{Group: "apps", Version: apiVersion, Kind: "Deployment"}, errs)
	testutil.AssertVectorCount(t, "apiserver_validation_declarative_validation_mismatch_total", nil, 0)
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testCases := map[string]struct {
				oldObj       apps.Deployment
				updateObj    apps.Deployment
				expectedErrs field.ErrorList
				skipVersions []string
				allRulesOnly bool
			}{
				"valid update": {
					oldObj:    mkValidDeployment(),
					updateObj: mkValidDeployment(),
				},
				"immutable selector": {
					oldObj: mkValidDeployment(),
					updateObj: mkValidDeployment(func(obj *apps.Deployment) {
						obj.Spec.Selector = &metav1.LabelSelector{
							MatchLabels: map[string]string{"app": "changed"},
						}
						obj.Spec.Template.Labels = map[string]string{"app": "changed"}
					}),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "selector"), &metav1.LabelSelector{
							MatchLabels: map[string]string{"app": "changed"},
						}, "field is immutable").WithOrigin("immutable").MarkAlpha(),
					},
				},
				"immutable selector set from unset": {
					oldObj:    mkValidDeployment(tweakSelector(nil)),
					updateObj: mkValidDeployment(),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "selector"), &metav1.LabelSelector{
							MatchLabels: map[string]string{"app": "test"},
						}, "field is immutable").WithOrigin("immutable").MarkAlpha(),
					},
				},
				"immutable selector unset from set": {
					oldObj:    mkValidDeployment(),
					updateObj: mkValidDeployment(tweakSelector(nil)),
					expectedErrs: field.ErrorList{
						field.Required(field.NewPath("spec", "selector"), "").MarkAlpha(),
						field.Invalid(field.NewPath("spec", "template", "metadata", "labels"), nil, "").MarkFromImperative(),
						field.Invalid(field.NewPath("spec", "selector"), nil, "field is immutable").WithOrigin("immutable").MarkAlpha(),
					},
					allRulesOnly: true,
				},
			}
			for name, tc := range testCases {
				t.Run(name, func(t *testing.T) {
					for _, version := range tc.skipVersions {
						if apiVersion == version {
							t.Skipf("not applicable to %s", apiVersion)
						}
					}
					ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
						APIPrefix:         "apis",
						APIGroup:          "apps",
						APIVersion:        apiVersion,
						Resource:          "deployments",
						Name:              "valid-deployment",
						IsResourceRequest: true,
						Verb:              "update",
					})
					tc.oldObj.ResourceVersion = "1"
					tc.updateObj.ResourceVersion = "2"
					if tc.allRulesOnly {
						verifyDeclarativeValidateUpdateAllRules(t, ctx, apiVersion, &tc.updateObj, &tc.oldObj, tc.expectedErrs)
						return
					}
					apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, registry.Strategy, tc.expectedErrs)
				})
			}
		})
	}
}

func verifyDeclarativeValidateUpdateAllRules(t *testing.T, ctx context.Context, apiVersion string, updateObj, oldObj *apps.Deployment, expectedErrs field.ErrorList) {
	t.Helper()

	testCtx := rest.WithAllDeclarativeEnforcedForTest(ctx)
	legacyregistry.Reset()
	defer legacyregistry.Reset()
	validationmetrics.ResetValidationMetricsInstance()
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.DeclarativeValidation:     true,
		features.DeclarativeValidationBeta: true,
	})

	errs := registry.Strategy.ValidateUpdate(testCtx, updateObj, oldObj)
	errs = registry.Strategy.ValidateDeclaratively(
		testCtx,
		updateObj,
		oldObj,
		errs,
		operation.Update,
		registry.Strategy.DeclarativeValidationConfig(testCtx, updateObj, oldObj),
	)

	field.ErrorMatcher{}.ByType().ByOrigin().ByField().ByValidationStabilityLevel().BySource().Test(t, expectedErrs, errs)
	coverage.RecordObservedRules(schema.GroupVersionKind{Group: "apps", Version: apiVersion, Kind: "Deployment"}, errs)
	testutil.AssertVectorCount(t, "apiserver_validation_declarative_validation_mismatch_total", nil, 0)
}

func mkValidDeployment(tweaks ...func(*apps.Deployment)) apps.Deployment {
	maxSurge := intstr.FromInt32(1)
	maxUnavailable := intstr.FromInt32(0)
	obj := apps.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-deployment",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: apps.DeploymentSpec{
			Replicas: 1,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": "test"},
			},
			Strategy: apps.DeploymentStrategy{
				Type: apps.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &apps.RollingUpdateDeployment{
					MaxSurge:       maxSurge,
					MaxUnavailable: maxUnavailable,
				},
			},
			Template: api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": "test"}},
				Spec:       mkValidPodSpec(),
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}

func mkValidPodSpec() api.PodSpec {
	return api.PodSpec{
		Containers: []api.Container{{
			Name:                     "test",
			Image:                    "test",
			ImagePullPolicy:          api.PullAlways,
			TerminationMessagePolicy: api.TerminationMessageReadFile,
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceCPU: resource.MustParse("100m"),
				},
			},
		}},
		RestartPolicy:                 api.RestartPolicyAlways,
		DNSPolicy:                     api.DNSClusterFirst,
		TerminationGracePeriodSeconds: ptr.To[int64](30),
	}
}

func tweakSelector(selector *metav1.LabelSelector) func(*apps.Deployment) {
	return func(obj *apps.Deployment) {
		obj.Spec.Selector = selector
	}
}

func tweakTolerations(tolerations ...api.Toleration) func(*apps.Deployment) {
	return func(obj *apps.Deployment) {
		obj.Spec.Template.Spec.Tolerations = tolerations
	}
}
