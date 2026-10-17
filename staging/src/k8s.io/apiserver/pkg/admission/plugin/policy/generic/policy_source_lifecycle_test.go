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

package generic_test

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission/plugin/policy/generic"
	"k8s.io/client-go/tools/cache"
)

// TestDirectFallbackResolvesParamAfterInformerLifecycle verifies that a named
// parameter remains resolvable after the last policy using its built-in kind is
// removed and a later policy receives the stopped shared informer from the
// factory. The informer remains stale, so this specifically exercises the
// direct API fallback rather than eventual watch convergence.
func TestDirectFallbackResolvesParamAfterInformerLifecycle(t *testing.T) {
	firstPolicy := &FakePolicy{
		ObjectMeta: metav1.ObjectMeta{Name: "first-policy"},
		ParamKind:  &admissionregistrationv1.ParamKind{APIVersion: "v1", Kind: "ConfigMap"},
	}
	firstBinding := &FakeBinding{
		ObjectMeta: metav1.ObjectMeta{Name: "first-binding"},
		PolicyName: "first-policy",
		ParamRef:   &admissionregistrationv1.ParamRef{Name: "first-param", Namespace: "repro"},
	}
	firstParam := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "first-param", Namespace: "repro"}}
	testContext, cancel, err := generic.NewPolicyTestContext(
		t,
		func(fp *FakePolicy) generic.PolicyAccessor { return fp },
		func(fb *FakeBinding) generic.BindingAccessor { return fb },
		func(*FakePolicy) generic.Evaluator { return nil },
		makeTestDispatcher,
		[]runtime.Object{firstPolicy, firstBinding, firstParam},
		nil,
	)
	require.NoError(t, err)
	defer cancel()
	require.NoError(t, testContext.Start())
	require.Len(t, testContext.Source.Hooks(), 1)
	firstInformer := testContext.Source.Hooks()[0].ParamInformer
	require.NotNil(t, firstInformer)
	ctx, stop := context.WithTimeout(testContext, 5*time.Second)
	defer stop()
	require.True(t, cache.WaitForCacheSync(ctx.Done(), firstInformer.Informer().HasSynced))
	require.NoError(t, testContext.WaitForReconcile(ctx, firstParam))
	require.NoError(t, testContext.DeleteAndWait(firstBinding, firstPolicy))
	require.Eventually(t, firstInformer.Informer().IsStopped, 5*time.Second, time.Millisecond)

	secondPolicy := &FakePolicy{
		ObjectMeta: metav1.ObjectMeta{Name: "second-policy"},
		ParamKind:  &admissionregistrationv1.ParamKind{APIVersion: "v1", Kind: "ConfigMap"},
	}
	secondBinding := &FakeBinding{
		ObjectMeta: metav1.ObjectMeta{Name: "second-binding"},
		PolicyName: "second-policy",
		ParamRef:   &admissionregistrationv1.ParamRef{Name: "second-param", Namespace: "repro"},
	}
	require.NoError(t, testContext.UpdateAndWait(secondPolicy, secondBinding))
	require.Len(t, testContext.Source.Hooks(), 1)
	hook := testContext.Source.Hooks()[0]
	require.True(t, hook.ParamInformer.Informer().IsStopped())
	require.True(t, hook.ParamInformer.Informer().HasSynced())

	secondParam := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "v1",
		"kind":       "ConfigMap",
		"metadata": map[string]interface{}{
			"name": "second-param", "namespace": "repro",
		},
		"data": map[string]interface{}{"identity": "expected"},
	}}
	_, err = testContext.DynamicClient.Resource(schema.GroupVersionResource{
		Group: "", Version: "v1", Resource: "configmaps",
	}).Namespace("repro").Create(ctx, secondParam, metav1.CreateOptions{})
	require.NoError(t, err)

	params, err := generic.CollectParams(
		secondPolicy.GetParamKind(), hook.ParamInformer, hook.ParamScope,
		secondBinding.GetParamRef(), "repro", hook.DynamicClient, hook.RESTMapper,
	)
	require.NoError(t, err)
	require.Len(t, params, 1)
	observed, ok := params[0].(*corev1.ConfigMap)
	require.Truef(t, ok, "expected typed ConfigMap, got %T", params[0])
	require.Equal(t, "expected", observed.Data["identity"])
}
