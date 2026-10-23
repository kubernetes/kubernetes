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

	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/apps"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	registry "k8s.io/kubernetes/pkg/registry/apps/deployment"
	poddeclarativevalidation "k8s.io/kubernetes/test/declarative_validation/core/pod"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

func tweakPodSpec(podTweaks ...podtest.Tweak) func(*apps.Deployment) {
	return func(deploy *apps.Deployment) {
		pod := &api.Pod{
			Spec: deploy.Spec.Template.Spec,
		}
		for _, tweak := range podTweaks {
			tweak(pod)
		}
		deploy.Spec.Template.Spec = pod.Spec
	}
}

func mkDeployment(tweaks ...func(*apps.Deployment)) *apps.Deployment {
	deploy := &apps.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
		Spec: apps.DeploymentSpec{
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"name": "abc"}},
			Strategy: apps.DeploymentStrategy{
				Type: apps.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &apps.RollingUpdateDeployment{
					MaxSurge:       intstr.FromInt32(1),
					MaxUnavailable: intstr.FromInt32(1),
				},
			},
			Template: api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"name": "abc"}},
				Spec:       podtest.MakePodSpec(),
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(deploy)
	}
	return deploy
}

func mkSchedulingDeployment(scheduling *apps.DeploymentSchedulingConfiguration) *apps.Deployment {
	return mkDeployment(func(d *apps.Deployment) {
		d.Spec.Replicas = 4
		d.Spec.Strategy = apps.DeploymentStrategy{Type: apps.RecreateDeploymentStrategyType}
		d.Spec.Scheduling = scheduling
	})
}

func TestDeclarativeValidate(t *testing.T) {
	schedulingPath := field.NewPath("spec", "scheduling")
	for _, apiVersion := range apiVersions {
		ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
			APIGroup:   "apps",
			APIVersion: apiVersion,
		})
		type testCase struct {
			input                        *apps.Deployment
			enableWorkloadWithDeployment bool
			expectedErrs                 field.ErrorList
		}
		testCases := map[string]testCase{
			"valid": {
				input: mkDeployment(),
			},
			"valid with basic scheduling policy": {
				input: mkSchedulingDeployment(&apps.DeploymentSchedulingConfiguration{
					SchedulingPolicy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{
						Basic: &schedulingv1alpha3.WorkloadPodGroupBasicSchedulingPolicy{},
					},
				}),
				enableWorkloadWithDeployment: true,
			},
			"valid with gang scheduling policy": {
				input: mkSchedulingDeployment(&apps.DeploymentSchedulingConfiguration{
					SchedulingPolicy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{
						Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: new(int32(1))},
					},
				}),
				enableWorkloadWithDeployment: true,
			},
			"scheduling forbidden when feature gate disabled": {
				input: mkSchedulingDeployment(&apps.DeploymentSchedulingConfiguration{}),
				expectedErrs: field.ErrorList{
					field.Forbidden(schedulingPath, ""),
				},
			},
			"scheduling policy with neither basic nor gang": {
				input: mkSchedulingDeployment(&apps.DeploymentSchedulingConfiguration{
					SchedulingPolicy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{},
				}),
				enableWorkloadWithDeployment: true,
				expectedErrs: field.ErrorList{
					field.Invalid(schedulingPath.Child("schedulingPolicy"), nil, "").WithOrigin("union"),
				},
			},
			"gang minCount zero": {
				input: mkSchedulingDeployment(&apps.DeploymentSchedulingConfiguration{
					SchedulingPolicy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{
						Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: new(int32(0))},
					},
				}),
				enableWorkloadWithDeployment: true,
				expectedErrs: field.ErrorList{
					field.Invalid(schedulingPath.Child("schedulingPolicy", "gang", "minCount"), nil, "").WithOrigin("minimum"),
				},
			},
			"disruption mode with neither single nor all": {
				input: mkSchedulingDeployment(&apps.DeploymentSchedulingConfiguration{
					DisruptionMode: &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{},
				}),
				enableWorkloadWithDeployment: true,
				expectedErrs: field.ErrorList{
					field.Invalid(schedulingPath.Child("disruptionMode"), nil, "").WithOrigin("union"),
				},
			},
			"topology constraints with too many entries": {
				input: mkSchedulingDeployment(&apps.DeploymentSchedulingConfiguration{
					SchedulingConstraints: &schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints{
						Topology: []schedulingv1alpha3.TopologyConstraint{
							{Key: "topology.kubernetes.io/zone"},
							{Key: "topology.kubernetes.io/rack"},
						},
					},
				}),
				enableWorkloadWithDeployment: true,
				expectedErrs: field.ErrorList{
					field.TooMany(schedulingPath.Child("schedulingConstraints", "topology"), 2, 1).WithOrigin("maxItems"),
				},
			},
			"topology constraint with empty key": {
				input: mkSchedulingDeployment(&apps.DeploymentSchedulingConfiguration{
					SchedulingConstraints: &schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints{
						Topology: []schedulingv1alpha3.TopologyConstraint{{Key: ""}},
					},
				}),
				enableWorkloadWithDeployment: true,
				expectedErrs: field.ErrorList{
					field.Required(schedulingPath.Child("schedulingConstraints", "topology").Index(0).Child("key"), ""),
				},
			},
			"topology constraint with invalid key": {
				input: mkSchedulingDeployment(&apps.DeploymentSchedulingConfiguration{
					SchedulingConstraints: &schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints{
						Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "invalid key"}},
					},
				}),
				enableWorkloadWithDeployment: true,
				expectedErrs: field.ErrorList{
					field.Invalid(schedulingPath.Child("schedulingConstraints", "topology").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key"),
				},
			},
			"too many resource claims": {
				input: mkSchedulingDeployment(&apps.DeploymentSchedulingConfiguration{
					ResourceClaims: []schedulingv1alpha3.WorkloadPodGroupResourceClaim{
						{Name: "c1", ResourceClaimName: new("rc1")},
						{Name: "c2", ResourceClaimName: new("rc2")},
						{Name: "c3", ResourceClaimName: new("rc3")},
						{Name: "c4", ResourceClaimName: new("rc4")},
						{Name: "c5", ResourceClaimName: new("rc5")},
					},
				}),
				enableWorkloadWithDeployment: true,
				expectedErrs: field.ErrorList{
					field.TooMany(schedulingPath.Child("resourceClaims"), 5, 4).WithOrigin("maxItems"),
				},
			},
			"resource claim with duplicate entries": {
				input: mkSchedulingDeployment(&apps.DeploymentSchedulingConfiguration{
					ResourceClaims: []schedulingv1alpha3.WorkloadPodGroupResourceClaim{
						{Name: "claim", ResourceClaimName: new("rc-1")},
						{Name: "claim", ResourceClaimName: new("rc-2")},
					},
				}),
				enableWorkloadWithDeployment: true,
				expectedErrs: field.ErrorList{
					field.Duplicate(schedulingPath.Child("resourceClaims").Index(1), nil),
				},
			},
			"resource claim with neither name nor template": {
				input: mkSchedulingDeployment(&apps.DeploymentSchedulingConfiguration{
					ResourceClaims: []schedulingv1alpha3.WorkloadPodGroupResourceClaim{{Name: "claim"}},
				}),
				enableWorkloadWithDeployment: true,
				expectedErrs: field.ErrorList{
					field.Invalid(schedulingPath.Child("resourceClaims").Index(0), nil, "").WithOrigin("union"),
				},
			},
			"resource claim with invalid short name": {
				input: mkSchedulingDeployment(&apps.DeploymentSchedulingConfiguration{
					ResourceClaims: []schedulingv1alpha3.WorkloadPodGroupResourceClaim{
						{Name: "invalid name", ResourceClaimName: new("rc")},
					},
				}),
				enableWorkloadWithDeployment: true,
				expectedErrs: field.ErrorList{
					field.Invalid(schedulingPath.Child("resourceClaims").Index(0).Child("name"), nil, "").WithOrigin("format=k8s-short-name"),
				},
			},
			"resource claim with empty name": {
				input: mkSchedulingDeployment(&apps.DeploymentSchedulingConfiguration{
					ResourceClaims: []schedulingv1alpha3.WorkloadPodGroupResourceClaim{
						{ResourceClaimName: new("rc")},
					},
				}),
				enableWorkloadWithDeployment: true,
				expectedErrs: field.ErrorList{
					field.Required(schedulingPath.Child("resourceClaims").Index(0).Child("name"), ""),
				},
			},
			"resource claim with invalid resourceClaimName": {
				input: mkSchedulingDeployment(&apps.DeploymentSchedulingConfiguration{
					ResourceClaims: []schedulingv1alpha3.WorkloadPodGroupResourceClaim{
						{Name: "claim", ResourceClaimName: new("invalid name")},
					},
				}),
				enableWorkloadWithDeployment: true,
				expectedErrs: field.ErrorList{
					field.Invalid(schedulingPath.Child("resourceClaims").Index(0).Child("resourceClaimName"), nil, "").WithOrigin("format=k8s-long-name"),
				},
			},
			"resource claim with invalid resourceClaimTemplateName": {
				input: mkSchedulingDeployment(&apps.DeploymentSchedulingConfiguration{
					ResourceClaims: []schedulingv1alpha3.WorkloadPodGroupResourceClaim{
						{Name: "claim", ResourceClaimTemplateName: new("invalid name")},
					},
				}),
				enableWorkloadWithDeployment: true,
				expectedErrs: field.ErrorList{
					field.Invalid(schedulingPath.Child("resourceClaims").Index(0).Child("resourceClaimTemplateName"), nil, "").WithOrigin("format=k8s-long-name"),
				},
			},
			"valid toleration key": {
				input: mkDeployment(tweakPodSpec(podtest.SetTolerations(api.Toleration{Key: "example.com/valid-key", Operator: api.TolerationOpExists}))),
			},
			"valid toleration key without prefix": {
				input: mkDeployment(tweakPodSpec(podtest.SetTolerations(api.Toleration{Key: "simple-key", Operator: api.TolerationOpExists}))),
			},
			"invalid toleration key format": {
				input: mkDeployment(tweakPodSpec(podtest.SetTolerations(api.Toleration{Key: "invalid key", Operator: api.TolerationOpExists}))),
				expectedErrs: field.ErrorList{
					field.Invalid(field.NewPath("spec", "template", "spec", "tolerations").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key").MarkAlpha(),
				},
			},
			"activeDeadlineSeconds minimum boundary violation": {
				input: mkDeployment(func(d *apps.Deployment) {
					deadline := int64(0)
					d.Spec.Template.Spec.ActiveDeadlineSeconds = &deadline
				}),
				expectedErrs: field.ErrorList{
					field.Forbidden(field.NewPath("spec", "template", "spec", "activeDeadlineSeconds"), "activeDeadlineSeconds in Deployment is not Supported").MarkFromImperative(),
					field.Invalid(field.NewPath("spec", "template", "spec", "activeDeadlineSeconds"), int64(0), "").WithOrigin("minimum").MarkAlpha(),
				},
			},
			"activeDeadlineSeconds maximum boundary violation": {
				input: mkDeployment(func(d *apps.Deployment) {
					deadline := int64(2147483648)
					d.Spec.Template.Spec.ActiveDeadlineSeconds = &deadline
				}),
				expectedErrs: field.ErrorList{
					field.Forbidden(field.NewPath("spec", "template", "spec", "activeDeadlineSeconds"), "activeDeadlineSeconds in Deployment is not Supported").MarkFromImperative(),
					field.Invalid(field.NewPath("spec", "template", "spec", "activeDeadlineSeconds"), int64(2147483648), "").WithOrigin("maximum").MarkAlpha(),
				},
			},
		}

		for _, tc := range poddeclarativevalidation.InvalidPodSpecTestcases(field.NewPath("spec", "template", "spec")) {
			testCases["podspec:"+tc.ID] = testCase{
				input:        mkDeployment(tweakPodSpec(tc.Tweaks...)),
				expectedErrs: tc.ExpectedErrs,
			}
		}

		for k, tc := range testCases {
			if apiVersion != "v1" && tc.input.Spec.Scheduling != nil {
				continue
			}
			t.Run(k, func(t *testing.T) {
				if tc.input.Spec.Scheduling != nil {
					featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
						features.GenericWorkload:        tc.enableWorkloadWithDeployment,
						features.WorkloadWithDeployment: tc.enableWorkloadWithDeployment,
					})
				}
				apitesting.VerifyValidationEquivalence(t, ctx, tc.input, registry.Strategy, tc.expectedErrs,
					apitesting.WithOmittedFieldPaths(map[schema.GroupVersionKind][]string{
						{Group: "apps", Version: "v1beta1", Kind: "Deployment"}: {"spec.scheduling"},
						{Group: "apps", Version: "v1beta2", Kind: "Deployment"}: {"spec.scheduling"},
					}),
				)
			})
		}
		meta.RunObjectMetaTestCases(t, ctx, mkDeployment(), registry.Strategy,
			meta.WithStringentFinalizerValidation(),
		)
		poddeclarativevalidation.RunDeclarativeValidateEvictionRespondersTestCases(t, ctx, registry.Strategy, field.NewPath("spec", "template", "spec"), mkDeployment(), func(baseObj *apps.Deployment, responders []api.EvictionResponder, schedulingGroup *api.PodSchedulingGroup) {
			baseObj.Spec.Template.Spec.EvictionResponders = responders
			baseObj.Spec.Template.Spec.SchedulingGroup = schedulingGroup
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	schedulingPath := field.NewPath("spec", "scheduling")
	for _, apiVersion := range apiVersions {
		ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
			APIGroup:   "apps",
			APIVersion: apiVersion,
			Name:       "abc",
			Verb:       "update",
		})
		meta.RunObjectMetaUpdateTestCases(t, ctx, mkDeployment(), registry.Strategy,
			meta.WithStringentFinalizerValidation(),
		)
		if apiVersion != "v1" {
			continue
		}
		testCases := map[string]struct {
			oldScheduling *apps.DeploymentSchedulingConfiguration
			scheduling    *apps.DeploymentSchedulingConfiguration
			expectedErrs  field.ErrorList
		}{
			"adding scheduling after creation is immutable": {
				scheduling: &apps.DeploymentSchedulingConfiguration{},
				expectedErrs: field.ErrorList{
					field.Invalid(schedulingPath, nil, "").WithOrigin("update"),
				},
			},
			"adding policy is immutable": {
				oldScheduling: &apps.DeploymentSchedulingConfiguration{},
				scheduling: &apps.DeploymentSchedulingConfiguration{
					SchedulingPolicy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{
						Basic: &schedulingv1alpha3.WorkloadPodGroupBasicSchedulingPolicy{},
					},
				},
				expectedErrs: field.ErrorList{
					field.Invalid(schedulingPath.Child("schedulingPolicy"), nil, "").WithOrigin("update"),
				},
			},
			"changing constraints is immutable": {
				oldScheduling: &apps.DeploymentSchedulingConfiguration{
					SchedulingConstraints: &schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints{
						Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "topology.kubernetes.io/zone"}},
					},
				},
				scheduling: &apps.DeploymentSchedulingConfiguration{
					SchedulingConstraints: &schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints{
						Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "topology.kubernetes.io/rack"}},
					},
				},
				expectedErrs: field.ErrorList{
					field.Invalid(schedulingPath.Child("schedulingConstraints"), nil, "").WithOrigin("immutable"),
				},
			},
			"switching disruption mode is immutable": {
				oldScheduling: &apps.DeploymentSchedulingConfiguration{
					SchedulingPolicy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{
						Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: new(int32(1))},
					},
					DisruptionMode: &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{
						Single: &schedulingv1alpha3.WorkloadPodGroupSingleDisruptionMode{},
					},
				},
				scheduling: &apps.DeploymentSchedulingConfiguration{
					SchedulingPolicy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{
						Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: new(int32(1))},
					},
					DisruptionMode: &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{
						All: &schedulingv1alpha3.WorkloadPodGroupAllDisruptionMode{},
					},
				},
				expectedErrs: field.ErrorList{
					field.Invalid(schedulingPath.Child("disruptionMode"), nil, "").WithOrigin("immutable"),
				},
			},
			"changing a resource claim name is immutable": {
				oldScheduling: &apps.DeploymentSchedulingConfiguration{
					ResourceClaims: []schedulingv1alpha3.WorkloadPodGroupResourceClaim{
						{Name: "claim", ResourceClaimName: new("rc-a")},
					},
				},
				scheduling: &apps.DeploymentSchedulingConfiguration{
					ResourceClaims: []schedulingv1alpha3.WorkloadPodGroupResourceClaim{
						{Name: "claim", ResourceClaimName: new("rc-b")},
					},
				},
				expectedErrs: field.ErrorList{
					field.Invalid(schedulingPath.Child("resourceClaims"), nil, "").WithOrigin("immutable"),
				},
			},
			"gang minCount change is valid": {
				oldScheduling: &apps.DeploymentSchedulingConfiguration{
					SchedulingPolicy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{
						Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: new(int32(1))},
					},
				},
				scheduling: &apps.DeploymentSchedulingConfiguration{
					SchedulingPolicy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{
						Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: new(int32(2))},
					},
				},
			},
		}
		for name, tc := range testCases {
			t.Run(name, func(t *testing.T) {
				featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
					features.GenericWorkload:        true,
					features.WorkloadWithDeployment: true,
				})
				old := mkSchedulingDeployment(tc.oldScheduling)
				old.ResourceVersion = "1"
				update := old.DeepCopy()
				update.Spec.Scheduling = tc.scheduling
				apitesting.VerifyUpdateValidationEquivalence(t, ctx, update, old, registry.Strategy, tc.expectedErrs,
					apitesting.WithOmittedFieldPaths(map[schema.GroupVersionKind][]string{
						{Group: "apps", Version: "v1beta1", Kind: "Deployment"}: {"spec.scheduling"},
						{Group: "apps", Version: "v1beta2", Kind: "Deployment"}: {"spec.scheduling"},
					}),
				)
			})
		}
	}
}
