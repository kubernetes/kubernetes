/*
Copyright 2026 The Kubernetes Authors.

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

package mutating

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/managedfields"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/policy/generic"
	"k8s.io/apiserver/pkg/admission/plugin/policy/mutating/patch"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

func TestMutatingWritebackPreservesEquivalentGVKMutations(t *testing.T) {
	inputGVK := schema.GroupVersionKind{Group: "example.test", Version: "v1", Kind: "Widget"}
	firstGVK := schema.GroupVersionKind{Group: "example.test", Version: "v1alpha1", Kind: "Widget"}
	secondGVK := schema.GroupVersionKind{Group: "example.test", Version: "v1beta1", Kind: "Widget"}
	object, attrs, objectInterfaces := newWritebackAttributes(t, inputGVK, nil, "")

	first, err := admission.NewVersionedAttributes(attrs, firstGVK, objectInterfaces)
	require.NoError(t, err)
	require.NoError(t, syncVersionedAttributes(attrs, objectInterfaces, first, firstGVK))
	addWritebackLabel(t, first, "first")
	require.NoError(t, writeBackVersionedAttributes(objectInterfaces, first))

	second, err := admission.NewVersionedAttributes(attrs, secondGVK, objectInterfaces)
	require.NoError(t, err)
	require.NoError(t, syncVersionedAttributes(attrs, objectInterfaces, second, secondGVK))
	assertWritebackLabel(t, second, "first")
	addWritebackLabel(t, second, "second")
	require.NoError(t, writeBackVersionedAttributes(objectInterfaces, second))
	assertWritebackLabel(t, &admission.VersionedAttributes{VersionedObject: admission.NewLazyObject(object)}, "first", "second")
}

func TestDispatchInvocationsPreservesPrewarmedEquivalentGVKs(t *testing.T) {
	inputGVK := schema.GroupVersionKind{Group: "example.test", Version: "v1", Kind: "Widget"}
	firstGVK := schema.GroupVersionKind{Group: "example.test", Version: "v1alpha1", Kind: "Widget"}
	secondGVK := schema.GroupVersionKind{Group: "example.test", Version: "v1beta1", Kind: "Widget"}
	object, attrs, objectInterfaces := newWritebackAttributes(t, inputGVK, nil, "")
	first, err := admission.NewVersionedAttributes(attrs, firstGVK, objectInterfaces)
	require.NoError(t, err)
	second, err := admission.NewVersionedAttributes(attrs, secondGVK, objectInterfaces)
	require.NoError(t, err)
	policy := func(name string, kind schema.GroupVersionKind, label string) generic.PolicyInvocation[*Policy, *PolicyBinding, PolicyEvaluator] {
		return generic.PolicyInvocation[*Policy, *PolicyBinding, PolicyEvaluator]{
			Policy:  &Policy{ObjectMeta: metav1.ObjectMeta{Name: name}, Spec: admissionregistrationv1.MutatingAdmissionPolicySpec{Mutations: []admissionregistrationv1.Mutation{{}}}},
			Binding: &PolicyBinding{ObjectMeta: metav1.ObjectMeta{Name: name + "-binding"}}, Kind: kind,
			Resource: kind.GroupVersion().WithResource("widgets"), Evaluator: PolicyEvaluator{Mutators: []patch.Patcher{writebackLabelPatcher{label: label}}},
		}
	}
	dispatcher := &dispatcher{authz: writebackAuthorizer{}, typeConverterManager: writebackTypeConverterManager{}}
	_, statusErr := dispatcher.dispatchInvocations(context.Background(), attrs, objectInterfaces, &writebackAccessor{attrs: map[schema.GroupVersionKind]*admission.VersionedAttributes{firstGVK: first, secondGVK: second}}, []generic.PolicyInvocation[*Policy, *PolicyBinding, PolicyEvaluator]{policy("first", firstGVK, "first"), policy("second", secondGVK, "second")})
	require.Nil(t, statusErr)
	labels := object.GetLabels()
	require.Equal(t, "set", labels["first"])
	require.Equal(t, "set", labels["second"])
}

func TestDispatchInvocationsPreservesSameGVKMutations(t *testing.T) {
	gvk := schema.GroupVersionKind{Group: "example.test", Version: "v1", Kind: "Widget"}
	object, attrs, objectInterfaces := newWritebackAttributes(t, gvk, nil, "")
	versioned, err := admission.NewVersionedAttributes(attrs, gvk, objectInterfaces)
	require.NoError(t, err)
	dispatcher := &dispatcher{authz: writebackAuthorizer{}, typeConverterManager: writebackTypeConverterManager{}}
	_, statusErr := dispatcher.dispatchInvocations(context.Background(), attrs, objectInterfaces, &writebackAccessor{attrs: map[schema.GroupVersionKind]*admission.VersionedAttributes{gvk: versioned}}, []generic.PolicyInvocation[*Policy, *PolicyBinding, PolicyEvaluator]{
		writebackInvocation(gvk, "first", writebackLabelPatcher{label: "first"}),
		writebackInvocation(gvk, "second", writebackLabelPatcher{label: "second"}),
	})
	require.Nil(t, statusErr)
	labels := object.GetLabels()
	require.Equal(t, "set", labels["first"])
	require.Equal(t, "set", labels["second"])
}

func TestSyncVersionedAttributesConvertsOldObjectAndKeepsSubresource(t *testing.T) {
	gvk := schema.GroupVersionKind{Group: "example.test", Version: "v1", Kind: "Widget"}
	targetGVK := schema.GroupVersionKind{Group: "example.test", Version: "v1alpha1", Kind: "Widget"}
	old := newWritebackObject(gvk)
	old.SetLabels(map[string]string{"old": "yes"})
	_, attrs, objectInterfaces := newWritebackAttributes(t, gvk, old, "status")
	versioned, err := admission.NewVersionedAttributes(attrs, targetGVK, objectInterfaces)
	require.NoError(t, err)
	require.NoError(t, syncVersionedAttributes(attrs, objectInterfaces, versioned, targetGVK))
	oldLabels, _, err := unstructured.NestedStringMap(versioned.VersionedOldObject.Object().(*unstructured.Unstructured).Object, "metadata", "labels")
	require.NoError(t, err)
	require.Equal(t, "yes", oldLabels["old"])
	require.Equal(t, targetGVK, versioned.VersionedKind)
	require.Equal(t, "status", attrs.GetSubresource())
}

func TestSyncVersionedAttributesConvertsTypedObject(t *testing.T) {
	inputGVK := schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"}
	targetGVK := schema.GroupVersionKind{Group: "apps", Version: "v1alpha1", Kind: "Deployment"}
	scheme := runtime.NewScheme()
	require.NoError(t, appsv1.AddToScheme(scheme))
	scheme.AddKnownTypeWithName(targetGVK, &unstructured.Unstructured{})
	object := &appsv1.Deployment{TypeMeta: metav1.TypeMeta{APIVersion: inputGVK.GroupVersion().String(), Kind: inputGVK.Kind}, ObjectMeta: metav1.ObjectMeta{Name: "typed"}}
	attrs := admission.NewAttributesRecord(object, nil, inputGVK, "", "typed", inputGVK.GroupVersion().WithResource("deployments"), "", admission.Create, &metav1.CreateOptions{}, false, nil)
	objectInterfaces := admission.NewObjectInterfacesFromScheme(scheme)
	versioned, err := admission.NewVersionedAttributes(attrs, targetGVK, objectInterfaces)
	require.NoError(t, err)
	require.NoError(t, syncVersionedAttributes(attrs, objectInterfaces, versioned, targetGVK))
	require.IsType(t, &unstructured.Unstructured{}, versioned.VersionedObject.Object())
}

func TestWriteBackVersionedAttributesReturnsConversionErrorAfterDirtyMutation(t *testing.T) {
	gvk := schema.GroupVersionKind{Group: "example.test", Version: "v1", Kind: "Widget"}
	object, attrs, objectInterfaces := newWritebackAttributes(t, gvk, nil, "")
	versioned, err := admission.NewVersionedAttributes(attrs, gvk, objectInterfaces)
	require.NoError(t, err)
	addWritebackLabel(t, versioned, "dirty")
	failingInterfaces := &admission.RuntimeObjectInterfaces{
		ObjectCreater:            objectInterfaces.GetObjectCreater(),
		ObjectTyper:              objectInterfaces.GetObjectTyper(),
		ObjectDefaulter:          objectInterfaces.GetObjectDefaulter(),
		ObjectConvertor:          failingWritebackConvertor{ObjectConvertor: objectInterfaces.GetObjectConvertor()},
		EquivalentResourceMapper: objectInterfaces.GetEquivalentResourceMapper(),
	}
	err = writeBackVersionedAttributes(failingInterfaces, versioned)
	require.Error(t, err)
	require.Contains(t, err.Error(), "failed to convert object")
	require.Empty(t, object.GetLabels(), "the canonical admission object must remain unchanged on write-back failure")
}

func TestDispatchInvocationsConversionErrorLeavesCanonicalObjectUnchanged(t *testing.T) {
	gvk := schema.GroupVersionKind{Group: "example.test", Version: "v1", Kind: "Widget"}
	object, attrs, regularInterfaces := newWritebackAttributes(t, gvk, nil, "")
	versioned, err := admission.NewVersionedAttributes(attrs, gvk, regularInterfaces)
	require.NoError(t, err)
	failingInterfaces := &admission.RuntimeObjectInterfaces{ObjectCreater: regularInterfaces.GetObjectCreater(), ObjectTyper: regularInterfaces.GetObjectTyper(), ObjectDefaulter: regularInterfaces.GetObjectDefaulter(), ObjectConvertor: failingWritebackConvertor{ObjectConvertor: regularInterfaces.GetObjectConvertor()}, EquivalentResourceMapper: regularInterfaces.GetEquivalentResourceMapper()}
	invocation := writebackInvocation(gvk, "dirty", writebackLabelPatcher{label: "dirty"})
	dispatcher := &dispatcher{authz: writebackAuthorizer{}, typeConverterManager: writebackTypeConverterManager{}}
	_, statusErr := dispatcher.dispatchInvocations(context.Background(), attrs, failingInterfaces, &writebackAccessor{attrs: map[schema.GroupVersionKind]*admission.VersionedAttributes{gvk: versioned}}, []generic.PolicyInvocation[*Policy, *PolicyBinding, PolicyEvaluator]{invocation})
	require.NotNil(t, statusErr)
	require.Empty(t, object.GetLabels())
}

func TestMutationFailureAfterEarlierSuccessWritesBackEarlierMutation(t *testing.T) {
	gvk := schema.GroupVersionKind{Group: "example.test", Version: "v1", Kind: "Widget"}
	object, attrs, objectInterfaces := newWritebackAttributes(t, gvk, nil, "")
	versioned, err := admission.NewVersionedAttributes(attrs, gvk, objectInterfaces)
	require.NoError(t, err)
	policy := &Policy{ObjectMeta: metav1.ObjectMeta{Name: "policy"}, Spec: admissionregistrationv1.MutatingAdmissionPolicySpec{Mutations: []admissionregistrationv1.Mutation{{}, {}}}}
	invocation := generic.PolicyInvocation[*Policy, *PolicyBinding, PolicyEvaluator]{
		Policy: policy, Binding: &PolicyBinding{}, Kind: gvk, Resource: gvk.GroupVersion().WithResource("widgets"),
		Evaluator: PolicyEvaluator{Mutators: []patch.Patcher{writebackLabelPatcher{label: "success"}, writebackErrorPatcher{}}},
	}
	accessor := &writebackAccessor{attrs: map[schema.GroupVersionKind]*admission.VersionedAttributes{gvk: versioned}}
	dispatcher := &dispatcher{authz: writebackAuthorizer{}, typeConverterManager: writebackTypeConverterManager{}}
	policyErrors, statusErr := dispatcher.dispatchInvocations(context.Background(), attrs, objectInterfaces, accessor, []generic.PolicyInvocation[*Policy, *PolicyBinding, PolicyEvaluator]{invocation})
	require.Nil(t, statusErr)
	require.Len(t, policyErrors, 1)
	labels := object.GetLabels()
	require.Equal(t, "set", labels["success"])
}

func TestDispatchInvocationsCrossGVKFailureKeepsEarlierSuccess(t *testing.T) {
	inputGVK := schema.GroupVersionKind{Group: "example.test", Version: "v1", Kind: "Widget"}
	firstGVK := schema.GroupVersionKind{Group: "example.test", Version: "v1alpha1", Kind: "Widget"}
	secondGVK := schema.GroupVersionKind{Group: "example.test", Version: "v1beta1", Kind: "Widget"}
	object, attrs, objectInterfaces := newWritebackAttributes(t, inputGVK, nil, "")
	first, err := admission.NewVersionedAttributes(attrs, firstGVK, objectInterfaces)
	require.NoError(t, err)
	second, err := admission.NewVersionedAttributes(attrs, secondGVK, objectInterfaces)
	require.NoError(t, err)
	dispatcher := &dispatcher{authz: writebackAuthorizer{}, typeConverterManager: writebackTypeConverterManager{}}
	policyErrors, statusErr := dispatcher.dispatchInvocations(context.Background(), attrs, objectInterfaces, &writebackAccessor{attrs: map[schema.GroupVersionKind]*admission.VersionedAttributes{firstGVK: first, secondGVK: second}}, []generic.PolicyInvocation[*Policy, *PolicyBinding, PolicyEvaluator]{writebackInvocation(firstGVK, "first", writebackLabelPatcher{label: "first"}), writebackInvocation(secondGVK, "second", writebackErrorPatcher{})})
	require.Nil(t, statusErr)
	require.Len(t, policyErrors, 1)
	require.Equal(t, "set", object.GetLabels()["first"])
}

func TestDispatchInvocationsReinvocationNoopNeverAndIfNeeded(t *testing.T) {
	for _, reinvocation := range []admissionregistrationv1.ReinvocationPolicyType{admissionregistrationv1.NeverReinvocationPolicy, admissionregistrationv1.IfNeededReinvocationPolicy} {
		t.Run(string(reinvocation), func(t *testing.T) {
			gvk := schema.GroupVersionKind{Group: "example.test", Version: "v1", Kind: "Widget"}
			object, attrs, objectInterfaces := newWritebackAttributes(t, gvk, nil, "")
			versioned, err := admission.NewVersionedAttributes(attrs, gvk, objectInterfaces)
			require.NoError(t, err)
			counter := 0
			invocation := writebackInvocationWithPolicy(gvk, "policy", reinvocation, writebackCountingPatcher{counter: &counter})
			dispatcher := &dispatcher{authz: writebackAuthorizer{}, typeConverterManager: writebackTypeConverterManager{}}
			accessor := &writebackAccessor{attrs: map[schema.GroupVersionKind]*admission.VersionedAttributes{gvk: versioned}}
			trigger := writebackInvocationWithPolicy(gvk, "trigger", admissionregistrationv1.NeverReinvocationPolicy, writebackLabelPatcher{label: "trigger"})
			_, statusErr := dispatcher.dispatchInvocations(context.Background(), attrs, objectInterfaces, accessor, []generic.PolicyInvocation[*Policy, *PolicyBinding, PolicyEvaluator]{invocation, trigger})
			require.Nil(t, statusErr)
			attrs.GetReinvocationContext().SetIsReinvoke()
			_, statusErr = dispatcher.dispatchInvocations(context.Background(), attrs, objectInterfaces, accessor, []generic.PolicyInvocation[*Policy, *PolicyBinding, PolicyEvaluator]{invocation, trigger})
			require.Nil(t, statusErr)
			if reinvocation == admissionregistrationv1.IfNeededReinvocationPolicy {
				require.Equal(t, 2, counter)
			} else {
				require.Equal(t, 1, counter)
			}
			_ = object
		})
	}

	gvk := schema.GroupVersionKind{Group: "example.test", Version: "v1", Kind: "Widget"}
	_, attrs, objectInterfaces := newWritebackAttributes(t, gvk, nil, "")
	versioned, err := admission.NewVersionedAttributes(attrs, gvk, objectInterfaces)
	require.NoError(t, err)
	invocation := writebackInvocationWithPolicy(gvk, "noop", admissionregistrationv1.IfNeededReinvocationPolicy, writebackNoopPatcher{})
	dispatcher := &dispatcher{authz: writebackAuthorizer{}, typeConverterManager: writebackTypeConverterManager{}}
	_, statusErr := dispatcher.dispatchInvocations(context.Background(), attrs, objectInterfaces, &writebackAccessor{attrs: map[schema.GroupVersionKind]*admission.VersionedAttributes{gvk: versioned}}, []generic.PolicyInvocation[*Policy, *PolicyBinding, PolicyEvaluator]{invocation})
	require.Nil(t, statusErr)
	require.False(t, attrs.GetReinvocationContext().ShouldReinvoke())
}

func newWritebackAttributes(t *testing.T, gvk schema.GroupVersionKind, old runtime.Object, subresource string) (*unstructured.Unstructured, admission.Attributes, admission.ObjectInterfaces) {
	t.Helper()
	scheme := runtime.NewScheme()
	scheme.AddKnownTypeWithName(gvk, &unstructured.Unstructured{})
	for _, version := range []string{"v1alpha1", "v1beta1"} {
		equivalentGVK := gvk
		equivalentGVK.Version = version
		scheme.AddKnownTypeWithName(equivalentGVK, &unstructured.Unstructured{})
	}
	object := newWritebackObject(gvk)
	attrs := admission.NewAttributesRecord(object, old, gvk, "", "demo", gvk.GroupVersion().WithResource("widgets"), subresource, admission.Create, &metav1.CreateOptions{}, false, nil)
	return object, attrs, admission.NewObjectInterfacesFromScheme(scheme)
}

func newWritebackObject(gvk schema.GroupVersionKind) *unstructured.Unstructured {
	object := &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": gvk.GroupVersion().String(), "kind": gvk.Kind, "metadata": map[string]interface{}{"name": "demo", "labels": map[string]interface{}{}}}}
	object.SetGroupVersionKind(gvk)
	return object
}

func addWritebackLabel(t *testing.T, attr *admission.VersionedAttributes, label string) {
	t.Helper()
	object := attr.VersionedObject.Object().(*unstructured.Unstructured).DeepCopy()
	require.NoError(t, unstructured.SetNestedField(object.Object, "set", "metadata", "labels", label))
	attr.UpdateObject(object)
}

func assertWritebackLabel(t *testing.T, attr *admission.VersionedAttributes, labels ...string) {
	t.Helper()
	got, _, err := unstructured.NestedStringMap(attr.VersionedObject.Object().(*unstructured.Unstructured).Object, "metadata", "labels")
	require.NoError(t, err)
	for _, label := range labels {
		require.Equal(t, "set", got[label])
	}
}

type writebackAccessor struct {
	attrs map[schema.GroupVersionKind]*admission.VersionedAttributes
}

func (a *writebackAccessor) VersionedAttribute(gvk schema.GroupVersionKind) (*admission.VersionedAttributes, error) {
	attr, ok := a.attrs[gvk]
	if !ok {
		return nil, fmt.Errorf("missing prewarmed attributes for %s", gvk)
	}
	return attr, nil
}

type writebackTypeConverterManager struct{}

func (writebackTypeConverterManager) GetTypeConverter(schema.GroupVersionKind) managedfields.TypeConverter {
	return managedfields.NewDeducedTypeConverter()
}

func (writebackTypeConverterManager) Run(context.Context) {}

type writebackLabelPatcher struct{ label string }

func (p writebackLabelPatcher) Patch(_ context.Context, request patch.Request, _ int64) (runtime.Object, error) {
	object := request.VersionedAttributes.VersionedObject.Object().(*unstructured.Unstructured).DeepCopy()
	if err := unstructured.SetNestedField(object.Object, "set", "metadata", "labels", p.label); err != nil {
		return nil, err
	}
	return object, nil
}

type writebackErrorPatcher struct{}

func (writebackErrorPatcher) Patch(context.Context, patch.Request, int64) (runtime.Object, error) {
	return nil, errors.New("synthetic mutation failure")
}

func writebackInvocation(gvk schema.GroupVersionKind, name string, patcher patch.Patcher) generic.PolicyInvocation[*Policy, *PolicyBinding, PolicyEvaluator] {
	return writebackInvocationWithPolicy(gvk, name, admissionregistrationv1.NeverReinvocationPolicy, patcher)
}

func writebackInvocationWithPolicy(gvk schema.GroupVersionKind, name string, reinvocation admissionregistrationv1.ReinvocationPolicyType, patcher patch.Patcher) generic.PolicyInvocation[*Policy, *PolicyBinding, PolicyEvaluator] {
	return generic.PolicyInvocation[*Policy, *PolicyBinding, PolicyEvaluator]{
		Policy:  &Policy{ObjectMeta: metav1.ObjectMeta{Name: name}, Spec: admissionregistrationv1.MutatingAdmissionPolicySpec{ReinvocationPolicy: reinvocation, Mutations: []admissionregistrationv1.Mutation{{}}}},
		Binding: &PolicyBinding{ObjectMeta: metav1.ObjectMeta{Name: name + "-binding"}}, Kind: gvk,
		Resource: gvk.GroupVersion().WithResource("widgets"), Evaluator: PolicyEvaluator{Mutators: []patch.Patcher{patcher}},
	}
}

type writebackCountingPatcher struct{ counter *int }

func (p writebackCountingPatcher) Patch(_ context.Context, request patch.Request, _ int64) (runtime.Object, error) {
	(*p.counter)++
	object := request.VersionedAttributes.VersionedObject.Object().(*unstructured.Unstructured).DeepCopy()
	if err := unstructured.SetNestedField(object.Object, "set", "metadata", "labels", "counted"); err != nil {
		return nil, err
	}
	return object, nil
}

type writebackNoopPatcher struct{}

func (writebackNoopPatcher) Patch(_ context.Context, request patch.Request, _ int64) (runtime.Object, error) {
	return request.VersionedAttributes.VersionedObject.Object().DeepCopyObject(), nil
}

type failingWritebackConvertor struct{ runtime.ObjectConvertor }

func (failingWritebackConvertor) Convert(interface{}, interface{}, interface{}) error {
	return errors.New("synthetic conversion failure")
}

var _ authorizer.UnconditionalAuthorizer = writebackAuthorizer{}

type writebackAuthorizer struct{}

func (writebackAuthorizer) Authorize(context.Context, authorizer.Attributes) (authorizer.Decision, string, error) {
	return authorizer.DecisionAllow, "test", nil
}
