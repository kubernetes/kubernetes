/*
Copyright 2018 The Kubernetes Authors.

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

package testing

import (
	"math/rand"
	"testing"

	admissionv1beta1 "k8s.io/api/admission/v1beta1"
	admissionregv1alpha1 "k8s.io/api/admissionregistration/v1alpha1"
	admissionregv1beta1 "k8s.io/api/admissionregistration/v1beta1"
	appsv1 "k8s.io/api/apps/v1"
	appsv1beta1 "k8s.io/api/apps/v1beta1"
	appsv1beta2 "k8s.io/api/apps/v1beta2"
	authenticationv1 "k8s.io/api/authentication/v1"
	authenticationv1beta1 "k8s.io/api/authentication/v1beta1"
	authorizationv1 "k8s.io/api/authorization/v1"
	authorizationv1beta1 "k8s.io/api/authorization/v1beta1"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	autoscalingv2beta1 "k8s.io/api/autoscaling/v2beta1"
	batchv1 "k8s.io/api/batch/v1"
	batchv1beta1 "k8s.io/api/batch/v1beta1"
	batchv2alpha1 "k8s.io/api/batch/v2alpha1"
	certificatesv1beta1 "k8s.io/api/certificates/v1beta1"
	corev1 "k8s.io/api/core/v1"
	eventsv1beta1 "k8s.io/api/events/v1beta1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	imagepolicyv1alpha1 "k8s.io/api/imagepolicy/v1alpha1"
	networkingv1 "k8s.io/api/networking/v1"
	policyv1beta1 "k8s.io/api/policy/v1beta1"
	rbacv1 "k8s.io/api/rbac/v1"
	rbacv1alpha1 "k8s.io/api/rbac/v1alpha1"
	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	schedulingv1alpha1 "k8s.io/api/scheduling/v1alpha1"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	settingsv1alpha1 "k8s.io/api/settings/v1alpha1"
	storagev1 "k8s.io/api/storage/v1"
	storagev1alpha1 "k8s.io/api/storage/v1alpha1"
	storagev1beta1 "k8s.io/api/storage/v1beta1"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/testing/fuzzer"
	"k8s.io/apimachinery/pkg/api/testing/roundtrip"
	genericfuzzer "k8s.io/apimachinery/pkg/apis/meta/fuzzer"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
)

var groups = []runtime.SchemeBuilder{
	admissionv1beta1.SchemeBuilder,
	admissionregv1alpha1.SchemeBuilder,
	admissionregv1beta1.SchemeBuilder,
	appsv1beta1.SchemeBuilder,
	appsv1beta2.SchemeBuilder,
	appsv1.SchemeBuilder,
	authenticationv1beta1.SchemeBuilder,
	authenticationv1.SchemeBuilder,
	authorizationv1beta1.SchemeBuilder,
	authorizationv1.SchemeBuilder,
	autoscalingv1.SchemeBuilder,
	autoscalingv2beta1.SchemeBuilder,
	batchv2alpha1.SchemeBuilder,
	batchv1beta1.SchemeBuilder,
	batchv1.SchemeBuilder,
	certificatesv1beta1.SchemeBuilder,
	corev1.SchemeBuilder,
	eventsv1beta1.SchemeBuilder,
	extensionsv1beta1.SchemeBuilder,
	imagepolicyv1alpha1.SchemeBuilder,
	networkingv1.SchemeBuilder,
	policyv1beta1.SchemeBuilder,
	rbacv1alpha1.SchemeBuilder,
	rbacv1beta1.SchemeBuilder,
	rbacv1.SchemeBuilder,
	schedulingv1alpha1.SchemeBuilder,
	schedulingv1beta1.SchemeBuilder,
	settingsv1alpha1.SchemeBuilder,
	storagev1alpha1.SchemeBuilder,
	storagev1beta1.SchemeBuilder,
	storagev1.SchemeBuilder,
}

func TestRoundTripExternalTypes(t *testing.T) {
	for _, builder := range groups {
		scheme := runtime.NewScheme()
		codecs := serializer.NewCodecFactory(scheme)

		require.NoError(t, builder.AddToScheme(scheme))
		seed := rand.Int63()
		// I'm only using the generic fuzzer funcs, but at some point in time we might need to
		// switch to specialized. For now we're happy with the current serialization test.
		fuzzer := fuzzer.FuzzerFor(genericfuzzer.Funcs, rand.NewSource(seed), codecs)

		roundtrip.RoundTripExternalTypes(t, scheme, codecs, fuzzer, nil)
	}
}

func TestFailRoundTrip(t *testing.T) {
	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)
	groupVersion := schema.GroupVersion{Group: "broken", Version: "v1"}
	builder := runtime.NewSchemeBuilder(func(scheme *runtime.Scheme) error {
		scheme.AddKnownTypes(groupVersion, &BrokenType{})
		metav1.AddToGroupVersion(scheme, groupVersion)
		return nil
	})
	require.NoError(t, builder.AddToScheme(scheme))
	seed := rand.Int63()
	fuzzer := fuzzer.FuzzerFor(genericfuzzer.Funcs, rand.NewSource(seed), codecs)
	tmpT := new(testing.T)
	roundtrip.RoundTripExternalTypes(tmpT, scheme, codecs, fuzzer, nil)
	// It's very hacky way of making sure the DeepCopy is actually invoked inside RoundTripExternalTypes
	// used in the other test. If for some reason this tests starts passing we need to fail b/c we're not testing
	// the DeepCopy in the other method which we care so much about.
	if !tmpT.Failed() {
		t.Log("RoundTrip should've failed on DeepCopy but it did not!")
		t.FailNow()
	}
}

type BrokenType struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Field1 string `json:"field1,omitempty"`
	Field2 string `json:"field2,omitempty"`
}

func (in *BrokenType) DeepCopy() *BrokenType {
	return new(BrokenType)
}

func (in *BrokenType) DeepCopyObject() runtime.Object {
	return in.DeepCopy()
}
