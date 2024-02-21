/*
Copyright 2024 The Kubernetes Authors.

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

package mutating_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/api/admissionregistration/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/managedfields"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/policy/generic"
	"k8s.io/apiserver/pkg/admission/plugin/policy/matching"
	"k8s.io/apiserver/pkg/admission/plugin/policy/mutating"
	"k8s.io/apiserver/pkg/admission/plugin/policy/mutating/patch"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/openapi/openapitest"
	"k8s.io/utils/ptr"
)

func setupTest(
	t *testing.T,
	compiler func(*mutating.Policy) mutating.PolicyEvaluator,
) *generic.PolicyTestContext[*mutating.Policy, *mutating.PolicyBinding, mutating.PolicyEvaluator] {

	testContext, testCancel, err := generic.NewPolicyTestContext[*mutating.Policy, *mutating.PolicyBinding, mutating.PolicyEvaluator](
		mutating.NewMutatingAdmissionPolicyAccessor,
		mutating.NewMutatingAdmissionPolicyBindingAccessor,
		compiler,
		func(a authorizer.Authorizer, m *matching.Matcher, i kubernetes.Interface) generic.Dispatcher[mutating.PolicyHook] {
			// Use embedded schemas rather than discovery schemas
			return mutating.NewDispatcher(a, m, patch.NewTypeConverterManager(nil, openapitest.NewEmbeddedFileClient()))
		},
		nil,
		[]meta.RESTMapping{
			{
				Resource: schema.GroupVersionResource{
					Group:    "",
					Version:  "v1",
					Resource: "pods",
				},
				GroupVersionKind: schema.GroupVersionKind{
					Group:   "",
					Version: "v1",
					Kind:    "Pod",
				},
				Scope: meta.RESTScopeNamespace,
			},
		})
	require.NoError(t, err)
	t.Cleanup(testCancel)
	require.NoError(t, testContext.Start())
	return testContext
}

// Show that a compiler that always sets an annotation on the object works
func TestBasicPatch(t *testing.T) {
	expectedAnnotations := map[string]string{"foo": "bar"}

	// Treat all policies as setting foo annotation to bar
	testContext := setupTest(t, func(p *mutating.Policy) mutating.PolicyEvaluator {
		return []mutating.MutationEvaluationFunc{func(
			ctx context.Context,
			matchedResource schema.GroupVersionResource,
			versionedAttr *admission.VersionedAttributes,
			o admission.ObjectInterfaces,
			versionedParams runtime.Object,
			namespace *corev1.Namespace,
			tc managedfields.TypeConverter,
			runtimeCELCostBudget int64,
		) (runtime.Object, error) {
			obj := versionedAttr.VersionedObject.DeepCopyObject()
			accessor, err := meta.Accessor(obj)
			if err != nil {
				return nil, err
			}
			accessor.SetAnnotations(expectedAnnotations)
			return obj, nil
		}}
	})

	// Set up a policy and binding that match, no params
	require.NoError(t, testContext.UpdateAndWait(
		&mutating.Policy{
			ObjectMeta: metav1.ObjectMeta{Name: "policy"},
			Spec: v1alpha1.MutatingAdmissionPolicySpec{
				MatchConstraints: &v1alpha1.MatchResources{
					MatchPolicy:       ptr.To(v1alpha1.Equivalent),
					NamespaceSelector: &metav1.LabelSelector{},
					ObjectSelector:    &metav1.LabelSelector{},
				},
				Mutations: []v1alpha1.Mutation{
					{
						Expression: "ignored, but required",
					},
				},
			},
		},
		&mutating.PolicyBinding{
			ObjectMeta: metav1.ObjectMeta{Name: "binding"},
			Spec: v1alpha1.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "policy",
			},
		},
	))

	// Show that if we run an object through the policy, it gets the annotation
	testObject := &corev1.ConfigMap{}
	err := testContext.Dispatch(testObject, nil, admission.Create)
	require.NoError(t, err)
	require.Equal(t, expectedAnnotations, testObject.Annotations)
}

func TestSSAPatch(t *testing.T) {
	patchObj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ConfigMap",
			"metadata": map[string]interface{}{
				"annotations": map[string]interface{}{
					"foo": "bar",
				},
			},
			"data": map[string]interface{}{
				"myfield": "myvalue",
			},
		},
	}

	testContext := setupTest(t, func(p *mutating.Policy) mutating.PolicyEvaluator {
		return []mutating.MutationEvaluationFunc{func(
			ctx context.Context,
			matchedResource schema.GroupVersionResource,
			versionedAttr *admission.VersionedAttributes,
			o admission.ObjectInterfaces,
			versionedParams runtime.Object,
			namespace *corev1.Namespace,
			tc managedfields.TypeConverter,
			runtimeCELCostBudget int64,
		) (runtime.Object, error) {
			return patch.ApplySMD(
				tc,
				versionedAttr.VersionedObject,
				patchObj,
			)
		}}
	})

	// Set up a policy and binding that match, no params
	require.NoError(t, testContext.UpdateAndWait(
		&mutating.Policy{
			ObjectMeta: metav1.ObjectMeta{Name: "policy"},
			Spec: v1alpha1.MutatingAdmissionPolicySpec{
				MatchConstraints: &v1alpha1.MatchResources{
					MatchPolicy:       ptr.To(v1alpha1.Equivalent),
					NamespaceSelector: &metav1.LabelSelector{},
					ObjectSelector:    &metav1.LabelSelector{},
				},
				Mutations: []v1alpha1.Mutation{
					{
						Expression: "ignored, but required",
					},
				},
			},
		},
		&mutating.PolicyBinding{
			ObjectMeta: metav1.ObjectMeta{Name: "binding"},
			Spec: v1alpha1.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "policy",
			},
		},
	))

	// Show that if we run an object through the policy, it gets the annotation
	testObject := &corev1.ConfigMap{}
	err := testContext.Dispatch(testObject, nil, admission.Create)
	require.NoError(t, err)
	require.Equal(t, &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{"foo": "bar"},
		},
		Data: map[string]string{"myfield": "myvalue"},
	}, testObject)
}

func TestSSAMapListAtomicMap(t *testing.T) {
	patchObj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Pod",
			"metadata": map[string]interface{}{
				"annotations": map[string]interface{}{
					"foo": "bar",
				},
			},
			"spec": map[string]interface{}{
				"initContainers": []interface{}{
					map[string]interface{}{
						"name":  "injected-init-container",
						"image": "injected-image",
					},
				},
				// node selector is atomic, so should be replaced
				"nodeSelector": map[string]interface{}{
					"custom": "nodeselector",
				},
			},
		},
	}

	testContext := setupTest(t, func(p *mutating.Policy) mutating.PolicyEvaluator {
		return []mutating.MutationEvaluationFunc{func(
			ctx context.Context,
			matchedResource schema.GroupVersionResource,
			versionedAttr *admission.VersionedAttributes,
			o admission.ObjectInterfaces,
			versionedParams runtime.Object,
			namespace *corev1.Namespace,
			tc managedfields.TypeConverter,
			runtimeCELCostBudget int64,
		) (runtime.Object, error) {
			return patch.ApplySMD(
				tc,
				versionedAttr.VersionedObject,
				patchObj,
			)
		}}
	})

	// Set up a policy and binding that match, no params
	require.NoError(t, testContext.UpdateAndWait(
		&mutating.Policy{
			ObjectMeta: metav1.ObjectMeta{Name: "policy"},
			Spec: v1alpha1.MutatingAdmissionPolicySpec{
				MatchConstraints: &v1alpha1.MatchResources{
					MatchPolicy:       ptr.To(v1alpha1.Equivalent),
					NamespaceSelector: &metav1.LabelSelector{},
					ObjectSelector:    &metav1.LabelSelector{},
				},
				Mutations: []v1alpha1.Mutation{
					{
						Expression: "ignored, but required",
					},
				},
			},
		},
		&mutating.PolicyBinding{
			ObjectMeta: metav1.ObjectMeta{Name: "binding"},
			Spec: v1alpha1.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "policy",
			},
		},
	))

	// Show that if we run an object through the policy, it gets the annotation
	testObject := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{},
		Spec: corev1.PodSpec{
			InitContainers: []corev1.Container{
				{
					Name:  "init-container",
					Image: "image",
				},
			},
			NodeSelector: map[string]string{"original": "not customized"},
		},
	}
	err := testContext.Dispatch(testObject, nil, admission.Create)
	require.NoError(t, err)
	require.Equal(t, &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{"foo": "bar"},
		},
		Spec: corev1.PodSpec{
			InitContainers: []corev1.Container{
				{
					Name:  "init-container",
					Image: "image",
				},
				{
					Name:  "injected-init-container",
					Image: "injected-image",
				},
			},
			NodeSelector: map[string]string{"custom": "nodeselector"},
		},
	}, testObject)
}
