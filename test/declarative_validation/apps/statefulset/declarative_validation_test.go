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
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	apps "k8s.io/kubernetes/pkg/apis/apps"
	_ "k8s.io/kubernetes/pkg/apis/apps/install"
	core "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/registry/apps/statefulset"
	"k8s.io/kubernetes/test/declarative_validation/meta"
	"testing"
)

// TODO: remove this apiVersions variable once coverage tests are generated for this package.
var apiVersions = []string{"v1", "v1beta1", "v1beta2"}

// Helper function to create a baseline valid StatefulSet with optional tweaks
func mkStatefulSet(tweaks ...func(*apps.StatefulSet)) apps.StatefulSet {
	var terminationGracePeriodSeconds int64 = 30
	obj := apps.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-resource-name",
		},
		Spec: apps.StatefulSetSpec{
			PodManagementPolicy: apps.OrderedReadyPodManagement,
			UpdateStrategy: apps.StatefulSetUpdateStrategy{
				Type: apps.RollingUpdateStatefulSetStrategyType,
			},
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "b"}},
			Template: core.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"a": "b"},
				},
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Name:                     "test",
							Image:                    "test",
							TerminationMessagePolicy: core.TerminationMessageReadFile,
							ImagePullPolicy:          core.PullIfNotPresent,
						},
					},
					RestartPolicy:                 core.RestartPolicyAlways,
					DNSPolicy:                     core.DNSClusterFirst,
					TerminationGracePeriodSeconds: &terminationGracePeriodSeconds,
				},
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			strategy := statefulset.Strategy
			var namespace string
			if strategy.NamespaceScoped() {
				namespace = metav1.NamespaceDefault
			}
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "apps",
				APIVersion:        apiVersion,
				Resource:          "statefulsets",
				Namespace:         namespace,
				IsResourceRequest: true,
				Verb:              "create",
			})
			obj := mkStatefulSet(func(o *apps.StatefulSet) {
				o.Namespace = namespace
			})
			meta.RunObjectMetaTestCases(t, ctx, &obj, strategy)

			t.Run("volumeClaimTemplates_metadata_name_empty", func(t *testing.T) {
				objWithEmptyVCTName := mkStatefulSet(func(o *apps.StatefulSet) {
					o.Namespace = namespace
					o.Spec.VolumeClaimTemplates = []core.PersistentVolumeClaim{
						{
							ObjectMeta: metav1.ObjectMeta{
								Name: "",
							},
							Spec: core.PersistentVolumeClaimSpec{
								AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
								Resources: core.VolumeResourceRequirements{
									Requests: core.ResourceList{
										core.ResourceStorage: resource.MustParse("10G"),
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
				apitesting.VerifyValidationEquivalence(t, ctx, &objWithEmptyVCTName, strategy, expectedErrs)
			})
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			strategy := statefulset.Strategy
			var namespace string
			if strategy.NamespaceScoped() {
				namespace = metav1.NamespaceDefault
			}
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "apps",
				APIVersion:        apiVersion,
				Resource:          "statefulsets",
				Namespace:         namespace,
				IsResourceRequest: true,
				Verb:              "update",
			})
			obj := mkStatefulSet(func(o *apps.StatefulSet) {
				o.Namespace = namespace
			})
			meta.RunObjectMetaUpdateTestCases(t, ctx, &obj, strategy)

			t.Run("volumeClaimTemplates_metadata_name_empty", func(t *testing.T) {
				objWithEmptyVCTName := mkStatefulSet(func(o *apps.StatefulSet) {
					o.Namespace = namespace
					o.Spec.VolumeClaimTemplates = []core.PersistentVolumeClaim{
						{
							ObjectMeta: metav1.ObjectMeta{
								Name: "",
							},
							Spec: core.PersistentVolumeClaimSpec{
								AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
								Resources: core.VolumeResourceRequirements{
									Requests: core.ResourceList{
										core.ResourceStorage: resource.MustParse("10G"),
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
				// VerifyUpdateValidationEquivalence is expecting the new object and the old object.
				// Oh, the original code had apitesting.VerifyValidationEquivalence even for update?
				// Let's keep it exactly as it was:
				apitesting.VerifyValidationEquivalence(t, ctx, &objWithEmptyVCTName, strategy, expectedErrs)
			})
		})
	}
}
