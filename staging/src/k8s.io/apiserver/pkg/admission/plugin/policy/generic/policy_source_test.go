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

package generic_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/policy/generic"
	"k8s.io/apiserver/pkg/admission/plugin/policy/matching"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
)

type fakeDispatcher struct{}

func (fd *fakeDispatcher) Dispatch(context.Context, admission.Attributes, admission.ObjectInterfaces, []generic.PolicyHook[*FakePolicy, *FakeBinding, generic.Evaluator]) error {
	return nil
}
func (fd *fakeDispatcher) Start(context.Context) error {
	return nil
}

func makeTestDispatcher(authorizer.UnconditionalAuthorizer, *matching.Matcher, kubernetes.Interface) generic.Dispatcher[generic.PolicyHook[*FakePolicy, *FakeBinding, generic.Evaluator]] {
	return &fakeDispatcher{}
}

func TestPolicySourceHasSyncedEmpty(t *testing.T) {
	testContext, testCancel, err := generic.NewPolicyTestContext(
		t,
		func(fp *FakePolicy) generic.PolicyAccessor { return fp },
		func(fb *FakeBinding) generic.BindingAccessor { return fb },
		func(fp *FakePolicy) generic.Evaluator { return nil },
		makeTestDispatcher,
		nil,
		nil,
	)
	require.NoError(t, err)
	defer testCancel()
	require.NoError(t, testContext.Start())

	// Should be able to wait for cache sync
	require.True(t, cache.WaitForCacheSync(testContext.Done(), testContext.Source.HasSynced), "cache should sync after informer running")
}

func TestPolicySourceHasSyncedInitialList(t *testing.T) {
	// Create a list of fake policies
	initialObjects := []runtime.Object{
		&FakePolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "policy1",
			},
		},
		&FakeBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "binding1",
			},
			PolicyName: "policy1",
		},
	}

	testContext, testCancel, err := generic.NewPolicyTestContext(
		t,
		func(fp *FakePolicy) generic.PolicyAccessor { return fp },
		func(fb *FakeBinding) generic.BindingAccessor { return fb },
		func(fp *FakePolicy) generic.Evaluator { return nil },
		makeTestDispatcher,
		initialObjects,
		nil,
	)
	require.NoError(t, err)
	defer testCancel()
	require.NoError(t, testContext.Start())
	// Should be able to wait for cache sync
	require.Len(t, testContext.Source.Hooks(), 1, "should have one policy")
	require.NoError(t, testContext.UpdateAndWait(
		&FakePolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "policy2",
			},
		},
		&FakeBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "binding2",
			},
			PolicyName: "policy2",
		},
	))
	require.Len(t, testContext.Source.Hooks(), 2, "should have two policies")
	require.NoError(t, testContext.UpdateAndWait(
		&FakePolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "policy3",
			},
		},
		&FakeBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "binding3",
			},
			PolicyName: "policy3",
		},
		&FakePolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "policy2",
			},
			ParamKind: &v1.ParamKind{
				APIVersion: "policy.example.com/v1",
				Kind:       "FakeParam",
			},
		},
	))
	require.Len(t, testContext.Source.Hooks(), 3, "should have 3 policies")

}

func TestPolicySourceBindsToPolicies(t *testing.T) {
	// Create a list of fake policies
	initialObjects := []runtime.Object{
		&FakePolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "policy1",
			},
		},
		&FakeBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "binding1",
			},
			PolicyName: "policy1",
		},
	}

	testContext, testCancel, err := generic.NewPolicyTestContext(
		t,
		func(fp *FakePolicy) generic.PolicyAccessor { return fp },
		func(fb *FakeBinding) generic.BindingAccessor { return fb },
		func(fp *FakePolicy) generic.Evaluator { return nil },
		makeTestDispatcher,
		initialObjects,
		nil,
	)
	require.NoError(t, err)
	defer testCancel()
	require.NoError(t, testContext.Start())

	require.Len(t, testContext.Source.Hooks(), 1, "should have one policy")
	require.Len(t, testContext.Source.Hooks()[0].Bindings, 1, "should have one binding")
	require.Equal(t, "binding1", testContext.Source.Hooks()[0].Bindings[0].GetName(), "should have one binding")

	// Change the binding to another policy (policies without bindings should
	// be ignored, so it should remove the first
	require.NoError(t, testContext.UpdateAndWait(
		&FakePolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name: "policy2",
			},
		},
		&FakeBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: "binding1",
			},
			PolicyName: "policy2",
		}))
	require.Len(t, testContext.Source.Hooks(), 1, "should have one policy")
	require.Equal(t, "policy2", testContext.Source.Hooks()[0].Policy.GetName(), "policy name should be policy2")
	require.Len(t, testContext.Source.Hooks()[0].Bindings, 1, "should have one binding")
	require.Equal(t, "binding1", testContext.Source.Hooks()[0].Bindings[0].GetName(), "binding name should be binding1")

}

type FakePolicy struct {
	metav1.TypeMeta
	metav1.ObjectMeta

	ParamKind *v1.ParamKind
}

var _ generic.PolicyAccessor = &FakePolicy{}

type FakeBinding struct {
	metav1.TypeMeta
	metav1.ObjectMeta

	PolicyName string
	ParamRef   *v1.ParamRef
}

var _ generic.BindingAccessor = &FakeBinding{}

func (fp *FakePolicy) GetName() string {
	return fp.Name
}

func (fp *FakePolicy) GetNamespace() string {
	return fp.Namespace
}

func (fp *FakePolicy) GetParamKind() *v1.ParamKind {
	return fp.ParamKind
}

func (fb *FakePolicy) GetMatchConstraints() *v1.MatchResources {
	return nil
}

func (fb *FakePolicy) GetFailurePolicy() *v1.FailurePolicyType {
	return nil
}

func (fb *FakeBinding) GetName() string {
	return fb.Name
}

func (fb *FakeBinding) GetNamespace() string {
	return fb.Namespace
}

func (fb *FakeBinding) GetPolicyName() types.NamespacedName {
	return types.NamespacedName{
		Name: fb.PolicyName,
	}
}

func (fb *FakeBinding) GetMatchResources() *v1.MatchResources {
	return nil
}

func (fb *FakeBinding) GetParamRef() *v1.ParamRef {
	return fb.ParamRef
}

func (fp *FakePolicy) DeepCopyObject() runtime.Object {
	// totally fudged deepcopy
	newFP := &FakePolicy{}
	*newFP = *fp
	return newFP
}

func (fb *FakeBinding) DeepCopyObject() runtime.Object {
	// totally fudged deepcopy
	newFB := &FakeBinding{}
	*newFB = *fb
	return newFB
}

// TestCollectParamsHandlesCacheMiss tests the race condition where a param
// exists in the API server but hasn't propagated to the informer cache yet.
// This can happen when Policy, Binding, and Param (e.g., ConfigMap) resources
// are created in rapid succession. The test verifies that CollectParams falls
// back to a direct API call when the cache returns NotFound.
func TestCollectParamsHandlesCacheMiss(t *testing.T) {
	testContext, testCancel, err := generic.NewPolicyTestContext(
		t,
		func(fp *FakePolicy) generic.PolicyAccessor { return fp },
		func(fb *FakeBinding) generic.BindingAccessor { return fb },
		func(fp *FakePolicy) generic.Evaluator { return nil },
		makeTestDispatcher,
		nil,
		nil,
	)
	require.NoError(t, err)
	defer testCancel()
	require.NoError(t, testContext.Start())

	// Create a ConfigMap directly in the dynamic client (simulating etcd)
	// without going through the informer, to simulate the race condition
	configMap := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "ConfigMap",
			"metadata": map[string]interface{}{
				"name":      "test-param",
				"namespace": "default",
			},
			"data": map[string]interface{}{
				"maxReplicas": "3",
			},
		},
	}

	// Add ConfigMap directly to dynamic client's tracker (bypassing informer)
	// This simulates the state where the object exists in etcd but the
	// watch event hasn't reached the informer cache yet
	ctx := context.Background()
	_, err = testContext.DynamicClient.Resource(schema.GroupVersionResource{
		Group:    "",
		Version:  "v1",
		Resource: "configmaps",
	}).Namespace("default").Create(ctx, configMap, metav1.CreateOptions{})
	require.NoError(t, err)

	// Now create policy and binding that reference this ConfigMap
	policy := &FakePolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-policy",
		},
		ParamKind: &v1.ParamKind{
			APIVersion: "v1",
			Kind:       "ConfigMap",
		},
	}

	binding := &FakeBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-binding",
		},
		PolicyName: "test-policy",
		ParamRef: &v1.ParamRef{
			Name:      "test-param",
			Namespace: "default",
		},
	}

	// Create policy and binding through normal path (they'll be in informers)
	require.NoError(t, testContext.UpdateAndWait(policy, binding))

	// Get the param informer for ConfigMap
	hooks := testContext.Source.Hooks()
	require.Len(t, hooks, 1, "should have one policy hook")

	hook := hooks[0]
	require.NotNil(t, hook.ParamInformer, "param informer should be set")

	// Call CollectParams - this should succeed via client fallback
	// even though the ConfigMap isn't in the informer cache
	params, err := generic.CollectParams(
		policy.GetParamKind(),
		hook.ParamInformer,
		hook.ParamScope,
		binding.GetParamRef(),
		"default",
		hook.DynamicClient,
		hook.RESTMapper,
	)

	// With the client fallback fix: this should succeed
	require.NoError(t, err, "CollectParams should succeed by falling back to direct client Get")
	require.Len(t, params, 1, "should have retrieved the param")
	require.NotNil(t, params[0], "param should not be nil")

	// Ensure fallback to get returns the same representation the informer cache would have
	// returned for the same object.
	cmParam, ok := params[0].(*corev1.ConfigMap)
	require.Truef(t, ok, "expected fallback to return *corev1.ConfigMap, got %T", params[0])
	require.Empty(t, cmParam.APIVersion, "typed param should have empty apiVersion, like informer-cached objects")
	require.Empty(t, cmParam.Kind, "typed param should have empty kind, like informer-cached objects")
	require.Equal(t, "test-param", cmParam.Name)
	require.Equal(t, map[string]string{"maxReplicas": "3"}, cmParam.Data)

	// Without the client fallback fix: the above would fail with NotFound error
	// because the informer cache doesn't have the ConfigMap yet
}

// TestCollectParamsHandlesMissingCRD verifies that when a paramKind references
// a CR for a CRD that the RESTMapper does not know about, CollectParams treats
// the param as missing and routes the binding through ParameterNotFoundAction
// rather than returning an internal error.
func TestCollectParamsHandlesMissingCRD(t *testing.T) {
	testContext, testCancel, err := generic.NewPolicyTestContext(
		t,
		func(fp *FakePolicy) generic.PolicyAccessor { return fp },
		func(fb *FakeBinding) generic.BindingAccessor { return fb },
		func(fp *FakePolicy) generic.Evaluator { return nil },
		makeTestDispatcher,
		nil,
		nil,
	)
	require.NoError(t, err)
	defer testCancel()
	require.NoError(t, testContext.Start())

	// Bootstrap a hook with a known paramKind (ConfigMap) so we get a
	// non-nil ParamInformer/RESTMapper/DynamicClient. The CollectParams
	// calls below substitute a different paramKind that the RESTMapper
	// cannot resolve, which is the scenario under test.
	policy := &FakePolicy{
		ObjectMeta: metav1.ObjectMeta{Name: "test-policy"},
		ParamKind: &v1.ParamKind{
			APIVersion: "v1",
			Kind:       "ConfigMap",
		},
	}
	binding := &FakeBinding{
		ObjectMeta: metav1.ObjectMeta{Name: "test-binding"},
		PolicyName: "test-policy",
		ParamRef: &v1.ParamRef{
			Name:      "ignored",
			Namespace: "default",
		},
	}
	require.NoError(t, testContext.UpdateAndWait(policy, binding))

	hooks := testContext.Source.Hooks()
	require.Len(t, hooks, 1, "should have one policy hook")
	hook := hooks[0]
	require.NotNil(t, hook.ParamInformer, "param informer should be set")
	require.NotNil(t, hook.RESTMapper, "restMapper should be set")
	require.NotNil(t, hook.DynamicClient, "dynamicClient should be set")

	missingParamKind := &v1.ParamKind{
		APIVersion: "missing.example.com/v1",
		Kind:       "MissingCRD",
	}

	t.Run("no ParameterNotFoundAction returns no params and no error", func(t *testing.T) {
		paramRef := &v1.ParamRef{
			Name:      "some-param",
			Namespace: "default",
		}

		params, err := generic.CollectParams(
			missingParamKind,
			hook.ParamInformer,
			meta.RESTScopeNamespace,
			paramRef,
			"default",
			hook.DynamicClient,
			hook.RESTMapper,
		)

		require.NoError(t, err, "an unknown paramKind must not surface as an internal error")
		require.Empty(t, params, "no params should be returned when the paramKind cannot be resolved")
	})

	t.Run("ParameterNotFoundAction=Deny returns deny error", func(t *testing.T) {
		denyAction := v1.DenyAction
		paramRef := &v1.ParamRef{
			Name:                    "some-param",
			Namespace:               "default",
			ParameterNotFoundAction: &denyAction,
		}

		params, err := generic.CollectParams(
			missingParamKind,
			hook.ParamInformer,
			meta.RESTScopeNamespace,
			paramRef,
			"default",
			hook.DynamicClient,
			hook.RESTMapper,
		)

		require.Error(t, err, "Deny action should produce an error when the param is missing")
		require.Contains(t, err.Error(), "no params found")
		require.Empty(t, params)
	})

	t.Run("ParameterNotFoundAction=Allow returns no params and no error", func(t *testing.T) {
		allowAction := v1.AllowAction
		paramRef := &v1.ParamRef{
			Name:                    "some-param",
			Namespace:               "default",
			ParameterNotFoundAction: &allowAction,
		}

		params, err := generic.CollectParams(
			missingParamKind,
			hook.ParamInformer,
			meta.RESTScopeNamespace,
			paramRef,
			"default",
			hook.DynamicClient,
			hook.RESTMapper,
		)

		require.NoError(t, err, "Allow action should not produce an error when the param is missing")
		require.Empty(t, params)
	})
}
