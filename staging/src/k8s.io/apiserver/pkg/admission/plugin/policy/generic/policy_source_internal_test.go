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

package generic

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/admissionregistration/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/policy/matching"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/client-go/kubernetes"
)

type internalFakeDispatcher struct{}

func (d *internalFakeDispatcher) Dispatch(context.Context, admission.Attributes, admission.ObjectInterfaces, []PolicyHook[*internalFakePolicy, *internalFakeBinding, Evaluator]) error {
	return nil
}

func (d *internalFakeDispatcher) Start(context.Context) error {
	return nil
}

func makeInternalFakeDispatcher(authorizer.UnconditionalAuthorizer, *matching.Matcher, kubernetes.Interface) Dispatcher[PolicyHook[*internalFakePolicy, *internalFakeBinding, Evaluator]] {
	return &internalFakeDispatcher{}
}

type internalFakePolicy struct {
	metav1.TypeMeta
	metav1.ObjectMeta
	ParamKind *v1.ParamKind
}

func (p *internalFakePolicy) GetName() string                         { return p.Name }
func (p *internalFakePolicy) GetNamespace() string                    { return p.Namespace }
func (p *internalFakePolicy) GetParamKind() *v1.ParamKind             { return p.ParamKind }
func (p *internalFakePolicy) GetMatchConstraints() *v1.MatchResources { return nil }
func (p *internalFakePolicy) GetFailurePolicy() *v1.FailurePolicyType { return nil }
func (p *internalFakePolicy) DeepCopyObject() runtime.Object          { copy := *p; return &copy }

type internalFakeBinding struct {
	metav1.TypeMeta
	metav1.ObjectMeta
	PolicyName string
}

func (b *internalFakeBinding) GetName() string      { return b.Name }
func (b *internalFakeBinding) GetNamespace() string { return b.Namespace }
func (b *internalFakeBinding) GetPolicyName() types.NamespacedName {
	return types.NamespacedName{Name: b.PolicyName}
}
func (b *internalFakeBinding) GetParamRef() *v1.ParamRef             { return nil }
func (b *internalFakeBinding) GetMatchResources() *v1.MatchResources { return nil }
func (b *internalFakeBinding) DeepCopyObject() runtime.Object        { copy := *b; return &copy }

type fixedErrorRESTMapper struct {
	err error
}

func (m fixedErrorRESTMapper) KindFor(schema.GroupVersionResource) (schema.GroupVersionKind, error) {
	return schema.GroupVersionKind{}, m.err
}

func (m fixedErrorRESTMapper) KindsFor(schema.GroupVersionResource) ([]schema.GroupVersionKind, error) {
	return nil, m.err
}

func (m fixedErrorRESTMapper) ResourceFor(schema.GroupVersionResource) (schema.GroupVersionResource, error) {
	return schema.GroupVersionResource{}, m.err
}

func (m fixedErrorRESTMapper) ResourcesFor(schema.GroupVersionResource) ([]schema.GroupVersionResource, error) {
	return nil, m.err
}

func (m fixedErrorRESTMapper) RESTMapping(schema.GroupKind, ...string) (*meta.RESTMapping, error) {
	return nil, m.err
}

func (m fixedErrorRESTMapper) RESTMappings(schema.GroupKind, ...string) ([]*meta.RESTMapping, error) {
	return nil, m.err
}

func (m fixedErrorRESTMapper) ResourceSingularizer(string) (string, error) {
	return "", m.err
}

type fakeParamKindResolver struct {
	mapping *meta.RESTMapping
	err     error
	handles func(schema.GroupVersionKind) bool
}

func (r fakeParamKindResolver) Resolve(context.Context, schema.GroupVersionKind) (*meta.RESTMapping, error) {
	return r.mapping, r.err
}

func (r fakeParamKindResolver) Handles(paramKind schema.GroupVersionKind) bool {
	return r.handles == nil || r.handles(paramKind)
}

func (r fakeParamKindResolver) HasSynced() bool {
	return true
}

func (r fakeParamKindResolver) RegisterForChanges(func(schema.GroupKind)) func() {
	return func() {}
}

func TestResolveParamKindRESTMappingUsesInjectedResolverWhenRESTMapperIsStale(t *testing.T) {
	paramKind := schema.GroupVersionKind{Group: "params.example.com", Version: "v1", Kind: "ExampleParam"}
	expected := &meta.RESTMapping{
		Resource:         paramKind.GroupVersion().WithResource("exampleparams"),
		GroupVersionKind: paramKind,
		Scope:            meta.RESTScopeNamespace,
	}
	source := &policySource[runtime.Object, runtime.Object, Evaluator]{
		ctx: context.Background(),
		restMapper: fixedErrorRESTMapper{err: &meta.NoKindMatchError{
			GroupKind:        paramKind.GroupKind(),
			SearchedVersions: []string{paramKind.Version},
		}},
		paramKindResolver: fakeParamKindResolver{mapping: expected},
	}

	mapping, err := source.resolveParamKindRESTMappingLocked(paramKind)

	require.NoError(t, err)
	require.Equal(t, expected, mapping)
}

func TestResolveParamKindRESTMappingKeepsResolverErrorsFailClosed(t *testing.T) {
	paramKind := schema.GroupVersionKind{Group: "params.example.com", Version: "v1", Kind: "ExampleParam"}
	source := &policySource[runtime.Object, runtime.Object, Evaluator]{
		ctx: context.Background(),
		restMapper: fixedErrorRESTMapper{err: &meta.NoKindMatchError{
			GroupKind:        paramKind.GroupKind(),
			SearchedVersions: []string{paramKind.Version},
		}},
		paramKindResolver: fakeParamKindResolver{err: errors.New("unable to list CRDs")},
	}

	_, err := source.resolveParamKindRESTMappingLocked(paramKind)

	require.ErrorContains(t, err, "unable to list CRDs")
}

func TestResolveParamKindRESTMappingRejectsStalePositiveRESTMapping(t *testing.T) {
	paramKind := schema.GroupVersionKind{Group: "params.example.com", Version: "v1", Kind: "ExampleParam"}
	restMapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{paramKind.GroupVersion()})
	restMapper.AddSpecific(
		paramKind,
		paramKind.GroupVersion().WithResource("exampleparams"),
		paramKind.GroupVersion().WithResource("exampleparam"),
		meta.RESTScopeNamespace,
	)
	source := &policySource[runtime.Object, runtime.Object, Evaluator]{
		ctx:               context.Background(),
		restMapper:        restMapper,
		paramKindResolver: fakeParamKindResolver{},
	}

	_, err := source.resolveParamKindRESTMappingLocked(paramKind)

	require.ErrorContains(t, err, "failed to find resource referenced by paramKind")
}

func TestResolveParamKindRESTMappingUsesRESTMapperForUnclaimedKind(t *testing.T) {
	paramKind := schema.GroupVersionKind{Group: "params.example.com", Version: "v1", Kind: "ExampleParam"}
	expected := &meta.RESTMapping{
		Resource:         paramKind.GroupVersion().WithResource("aggregatedparams"),
		GroupVersionKind: paramKind,
		Scope:            meta.RESTScopeNamespace,
	}
	restMapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{paramKind.GroupVersion()})
	restMapper.AddSpecific(paramKind, expected.Resource, expected.Resource, expected.Scope)
	source := &policySource[runtime.Object, runtime.Object, Evaluator]{
		ctx:        context.Background(),
		restMapper: restMapper,
		paramKindResolver: fakeParamKindResolver{handles: func(schema.GroupVersionKind) bool {
			return false
		}},
	}

	mapping, err := source.resolveParamKindRESTMappingLocked(paramKind)

	require.NoError(t, err)
	require.Equal(t, expected, mapping)
}

func TestPolicySourcePublishesConfigurationErrorWhenParamKindCannotBeResolved(t *testing.T) {
	policy := &internalFakePolicy{
		ObjectMeta: metav1.ObjectMeta{Name: "test-policy"},
		ParamKind:  &v1.ParamKind{APIVersion: "params.example.com/v1", Kind: "ExampleParam"},
	}
	binding := &internalFakeBinding{
		ObjectMeta: metav1.ObjectMeta{Name: "test-binding"},
		PolicyName: "test-policy",
	}
	testContext, testCancel, err := NewPolicyTestContext(
		t,
		func(p *internalFakePolicy) PolicyAccessor { return p },
		func(b *internalFakeBinding) BindingAccessor { return b },
		func(*internalFakePolicy) Evaluator { return nil },
		makeInternalFakeDispatcher,
		[]runtime.Object{policy, binding},
		nil,
		fakeParamKindResolver{err: errors.New("unable to resolve CRD")},
	)
	require.NoError(t, err)
	defer testCancel()

	require.NoError(t, testContext.Start())

	require.Len(t, testContext.Source.Hooks(), 1)
	require.ErrorContains(t, testContext.Source.Hooks()[0].ConfigurationError, "unable to resolve CRD")
}

func TestPolicySourceMarksParamInformerForReconciliationWhenKindChanges(t *testing.T) {
	changedKind := schema.GroupVersionKind{Group: "params.example.com", Version: "v1", Kind: "ExampleParam"}
	unrelatedKind := schema.GroupVersionKind{Group: "other.example.com", Version: "v1", Kind: "OtherParam"}
	changedCanceled := false
	unrelatedCanceled := false
	source := &policySource[runtime.Object, runtime.Object, Evaluator]{
		paramsCRDControllers: map[schema.GroupVersionKind]*paramInfo{
			changedKind: {
				cancelFunc: func() { changedCanceled = true },
			},
			unrelatedKind: {
				cancelFunc: func() { unrelatedCanceled = true },
			},
		},
		paramKindsToReconcile: map[schema.GroupVersionKind]struct{}{},
	}

	source.paramKindChanged(changedKind.GroupKind())

	require.False(t, changedCanceled)
	require.False(t, unrelatedCanceled)
	require.Contains(t, source.paramsCRDControllers, changedKind)
	require.Contains(t, source.paramsCRDControllers, unrelatedKind)
	require.Contains(t, source.paramKindsToReconcile, changedKind)
	require.NotContains(t, source.paramKindsToReconcile, unrelatedKind)
	require.True(t, source.policiesDirty.Load())
}

func TestPolicySourceCleansReconciliationStateForOrphanedParamInformer(t *testing.T) {
	paramKind := schema.GroupVersionKind{Group: "params.example.com", Version: "v1", Kind: "ExampleParam"}
	testContext, testCancel, err := NewPolicyTestContext(
		t,
		func(p *internalFakePolicy) PolicyAccessor { return p },
		func(b *internalFakeBinding) BindingAccessor { return b },
		func(*internalFakePolicy) Evaluator { return nil },
		makeInternalFakeDispatcher,
		nil,
		nil,
		nil,
	)
	require.NoError(t, err)
	defer testCancel()
	require.NoError(t, testContext.Start())

	canceled := false
	source := testContext.Source.(*policySource[*internalFakePolicy, *internalFakeBinding, Evaluator])
	source.lock.Lock()
	source.paramsCRDControllers[paramKind] = &paramInfo{cancelFunc: func() { canceled = true }}
	source.paramKindsToReconcile[paramKind] = struct{}{}
	source.lock.Unlock()

	_, err = source.calculatePolicyData()

	require.NoError(t, err)
	require.True(t, canceled)
	require.NotContains(t, source.paramsCRDControllers, paramKind)
	require.NotContains(t, source.paramKindsToReconcile, paramKind)
}

func TestPolicySourcePreservesParamInformerWhenMappingIsUnchanged(t *testing.T) {
	paramKind := schema.GroupVersionKind{Group: "params.example.com", Version: "v1", Kind: "ExampleParam"}
	mapping := meta.RESTMapping{
		Resource:         paramKind.GroupVersion().WithResource("exampleparams"),
		GroupVersionKind: paramKind,
		Scope:            meta.RESTScopeNamespace,
	}
	canceled := false
	existingInfo := &paramInfo{
		mapping:    mapping,
		cancelFunc: func() { canceled = true },
	}
	source := &policySource[runtime.Object, runtime.Object, Evaluator]{
		ctx:                   context.Background(),
		paramKindResolver:     fakeParamKindResolver{mapping: &mapping},
		paramsCRDControllers:  map[schema.GroupVersionKind]*paramInfo{paramKind: existingInfo},
		paramKindsToReconcile: map[schema.GroupVersionKind]struct{}{},
	}
	source.paramKindChanged(paramKind.GroupKind())

	_, resolvedMapping, err := source.ensureParamsForPolicyLocked(&paramKind)

	require.NoError(t, err)
	require.Same(t, existingInfo, source.paramsCRDControllers[paramKind])
	require.Equal(t, &mapping, resolvedMapping)
	require.False(t, canceled)
	require.NotContains(t, source.paramKindsToReconcile, paramKind)
}

func TestPolicySourcePreservesUnclaimedParamInformerOnCRDChange(t *testing.T) {
	paramKind := schema.GroupVersionKind{Group: "params.example.com", Version: "v1", Kind: "ExampleParam"}
	mapping := meta.RESTMapping{
		Resource:         paramKind.GroupVersion().WithResource("aggregatedparams"),
		GroupVersionKind: paramKind,
		Scope:            meta.RESTScopeNamespace,
	}
	restMapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{paramKind.GroupVersion()})
	restMapper.AddSpecific(paramKind, mapping.Resource, mapping.Resource, mapping.Scope)
	canceled := false
	existingInfo := &paramInfo{
		mapping:    mapping,
		cancelFunc: func() { canceled = true },
	}
	source := &policySource[runtime.Object, runtime.Object, Evaluator]{
		ctx:        context.Background(),
		restMapper: restMapper,
		paramKindResolver: fakeParamKindResolver{handles: func(schema.GroupVersionKind) bool {
			return false
		}},
		paramsCRDControllers:  map[schema.GroupVersionKind]*paramInfo{paramKind: existingInfo},
		paramKindsToReconcile: map[schema.GroupVersionKind]struct{}{},
	}
	source.paramKindChanged(paramKind.GroupKind())

	_, resolvedMapping, err := source.ensureParamsForPolicyLocked(&paramKind)

	require.NoError(t, err)
	require.Same(t, existingInfo, source.paramsCRDControllers[paramKind])
	require.Equal(t, &mapping, resolvedMapping)
	require.False(t, canceled)
	require.NotContains(t, source.paramKindsToReconcile, paramKind)
}

func TestPolicySourceStopsParamInformerWhenMappingDisappears(t *testing.T) {
	paramKind := schema.GroupVersionKind{Group: "params.example.com", Version: "v1", Kind: "ExampleParam"}
	canceled := false
	source := &policySource[runtime.Object, runtime.Object, Evaluator]{
		ctx:               context.Background(),
		paramKindResolver: fakeParamKindResolver{},
		paramsCRDControllers: map[schema.GroupVersionKind]*paramInfo{
			paramKind: {
				mapping: meta.RESTMapping{
					Resource:         paramKind.GroupVersion().WithResource("exampleparams"),
					GroupVersionKind: paramKind,
					Scope:            meta.RESTScopeNamespace,
				},
				cancelFunc: func() { canceled = true },
			},
		},
		paramKindsToReconcile: map[schema.GroupVersionKind]struct{}{},
	}
	source.paramKindChanged(paramKind.GroupKind())

	_, _, err := source.ensureParamsForPolicyLocked(&paramKind)

	require.ErrorContains(t, err, "failed to find resource referenced by paramKind")
	require.True(t, canceled)
	require.NotContains(t, source.paramsCRDControllers, paramKind)
	require.NotContains(t, source.paramKindsToReconcile, paramKind)
}
