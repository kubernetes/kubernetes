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

package statefulset

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/apps"
	_ "k8s.io/kubernetes/pkg/apis/apps/install"
	api "k8s.io/kubernetes/pkg/apis/core"
	registry "k8s.io/kubernetes/pkg/registry/apps/statefulset"
	"k8s.io/kubernetes/test/declarative_validation/meta"
	"k8s.io/utils/ptr"
)

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "apps",
		APIVersion:        apiVersion,
		Resource:          "statefulsets",
		IsResourceRequest: true,
		Verb:              "create",
	})

	obj := mkValidStatefulSet()
	meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())

	t.Run("volumeClaimTemplates_metadata_name_empty", func(t *testing.T) {
		objWithEmptyVCTName := mkValidStatefulSet(func(o *apps.StatefulSet) {
			o.Spec.VolumeClaimTemplates = []api.PersistentVolumeClaim{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "",
					},
					Spec: api.PersistentVolumeClaimSpec{
						AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
						Resources: api.VolumeResourceRequirements{
							Requests: api.ResourceList{
								api.ResourceStorage: resource.MustParse("10G"),
							},
						},
					},
				},
			}
		})
		expectedErrs := field.ErrorList{
			field.Required(field.NewPath("spec", "template", "spec", "volumes").Index(0).Child("name"), "").MarkFromImperative(),
			field.Required(field.NewPath("spec", "template", "spec", "volumes").Index(0).Child("persistentVolumeClaim", "claimName"), "").MarkFromImperative(),
		}
		apitesting.VerifyValidationEquivalence(t, ctx, &objWithEmptyVCTName, registry.Strategy, expectedErrs)
	})

	testCases := map[string]struct {
		input apps.StatefulSet

		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidStatefulSet(),
		},
		"valid toleration key": {
			input: mkValidStatefulSet(tweakTolerations(api.Toleration{Key: "example.com/valid-key", Operator: api.TolerationOpExists})),
		},
		"valid toleration key without prefix": {
			input: mkValidStatefulSet(tweakTolerations(api.Toleration{Key: "simple-key", Operator: api.TolerationOpExists})),
		},
		"invalid toleration key format": {
			input: mkValidStatefulSet(tweakTolerations(api.Toleration{Key: "invalid key", Operator: api.TolerationOpExists})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "template", "spec", "tolerations").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key").MarkAlpha(),
			},
		},
		"selector required": {
			input: mkValidStatefulSet(tweakSelectorLabels(nil)),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec").Child("selector"), "").MarkAlpha(),
				field.Invalid(field.NewPath("spec").Child("template").Child("metadata").Child("labels"), nil, "").MarkFromImperative(),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, registry.Strategy, tc.expectedErrs)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {

	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "apps",
		APIVersion:        apiVersion,
		Resource:          "statefulsets",
		Name:              "valid-statefulset",
		IsResourceRequest: true,
		Verb:              "update",
	})
	obj := mkValidStatefulSet()
	meta.RunObjectMetaUpdateTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())

	testCases := map[string]struct {
		oldObj       apps.StatefulSet
		updateObj    apps.StatefulSet
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldObj:    mkValidStatefulSet(),
			updateObj: mkValidStatefulSet(),
		},
		"selector changed": {
			oldObj:    mkValidStatefulSet(),
			updateObj: mkValidStatefulSet(tweakSelectorLabels(map[string]string{"a": "c"})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("template").Child("metadata").Child("labels"), nil, "").MarkFromImperative(),
				field.Invalid(field.NewPath("spec").Child("selector"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"selector set from unset": {
			oldObj:    mkValidStatefulSet(tweakSelectorLabels(nil)),
			updateObj: mkValidStatefulSet(),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("selector"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"selector unset from set": {
			oldObj:    mkValidStatefulSet(),
			updateObj: mkValidStatefulSet(tweakSelectorLabels(map[string]string{})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("selector"), nil, "").MarkFromImperative(),
				field.Invalid(field.NewPath("spec").Child("selector"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"serviceName changed": {
			oldObj:    mkValidStatefulSet(),
			updateObj: mkValidStatefulSet(tweakServiceName("other-service")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("serviceName"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"serviceName set from unset": {
			oldObj:    mkValidStatefulSet(tweakServiceName("")),
			updateObj: mkValidStatefulSet(),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("serviceName"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"serviceName unset from set": {
			oldObj:    mkValidStatefulSet(),
			updateObj: mkValidStatefulSet(tweakServiceName("")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("serviceName"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"podManagementPolicy changed": {
			oldObj:    mkValidStatefulSet(),
			updateObj: mkValidStatefulSet(tweakPodManagementPolicy(apps.ParallelPodManagement)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("podManagementPolicy"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"podManagementPolicy set from unset": {
			oldObj:    mkValidStatefulSet(tweakPodManagementPolicy("")),
			updateObj: mkValidStatefulSet(),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("podManagementPolicy"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"podManagementPolicy unset from set": {
			oldObj:    mkValidStatefulSet(),
			updateObj: mkValidStatefulSet(tweakPodManagementPolicy("")),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec").Child("podManagementPolicy"), "").MarkFromImperative(),
				field.Invalid(field.NewPath("spec").Child("podManagementPolicy"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"volumeClaimTemplates changed": {
			oldObj:    mkValidStatefulSet(tweakVolumeClaimTemplate("pvc-abc", "1Gi")),
			updateObj: mkValidStatefulSet(tweakVolumeClaimTemplate("pvc-abc", "2Gi")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("volumeClaimTemplates"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"volumeClaimTemplates set from unset": {
			oldObj:    mkValidStatefulSet(),
			updateObj: mkValidStatefulSet(tweakVolumeClaimTemplate("pvc-abc", "1Gi")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("volumeClaimTemplates"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"volumeClaimTemplates unset from set": {
			oldObj:    mkValidStatefulSet(tweakVolumeClaimTemplate("pvc-abc", "1Gi")),
			updateObj: mkValidStatefulSet(),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("volumeClaimTemplates"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.oldObj.ResourceVersion = "1"
			tc.updateObj.ResourceVersion = "2"
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIPrefix:         "apis",
				APIGroup:          "apps",
				APIVersion:        apiVersion,
				Resource:          "statefulsets",
				Name:              "valid-statefulset",
				IsResourceRequest: true,
				Verb:              "update",
			})
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, registry.Strategy, tc.expectedErrs)
		})
	}
}

func mkValidStatefulSet(tweaks ...func(obj *apps.StatefulSet)) apps.StatefulSet {
	validSelector := map[string]string{"a": "b"}
	obj := apps.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-statefulset",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: apps.StatefulSetSpec{
			Replicas:            1,
			ServiceName:         "valid-service",
			PodManagementPolicy: apps.OrderedReadyPodManagement,
			Selector: &metav1.LabelSelector{
				MatchLabels: validSelector,
			},
			Template: api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: validSelector,
				},
				Spec: api.PodSpec{
					RestartPolicy:                 api.RestartPolicyAlways,
					DNSPolicy:                     api.DNSClusterFirst,
					Containers:                    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
					TerminationGracePeriodSeconds: ptr.To[int64](30),
				},
			},
			UpdateStrategy: apps.StatefulSetUpdateStrategy{
				Type: apps.RollingUpdateStatefulSetStrategyType,
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}

func tweakSelectorLabels(labels map[string]string) func(obj *apps.StatefulSet) {
	return func(obj *apps.StatefulSet) {
		if labels == nil {
			obj.Spec.Selector = nil
			return
		}
		obj.Spec.Selector = &metav1.LabelSelector{MatchLabels: labels}
	}
}

func tweakTolerations(tolerations ...api.Toleration) func(obj *apps.StatefulSet) {
	return func(obj *apps.StatefulSet) {
		obj.Spec.Template.Spec.Tolerations = tolerations
	}
}

func tweakServiceName(serviceName string) func(obj *apps.StatefulSet) {
	return func(obj *apps.StatefulSet) {
		obj.Spec.ServiceName = serviceName
	}
}

func tweakPodManagementPolicy(policy apps.PodManagementPolicyType) func(obj *apps.StatefulSet) {
	return func(obj *apps.StatefulSet) {
		obj.Spec.PodManagementPolicy = policy
	}
}

func tweakVolumeClaimTemplate(name, storage string) func(obj *apps.StatefulSet) {
	return func(obj *apps.StatefulSet) {
		obj.Spec.VolumeClaimTemplates = []api.PersistentVolumeClaim{{
			ObjectMeta: metav1.ObjectMeta{Name: name},
			Spec: api.PersistentVolumeClaimSpec{
				AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
				Resources: api.VolumeResourceRequirements{
					Requests: api.ResourceList{
						api.ResourceStorage: resource.MustParse(storage),
					},
				},
			},
		}}
	}
}
