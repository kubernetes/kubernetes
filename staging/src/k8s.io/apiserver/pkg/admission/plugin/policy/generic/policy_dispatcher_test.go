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
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/dynamic/dynamicinformer"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/utils/ptr"
)

// TestConvertParamToInformerRepresentation verifies that params fetched with the
// dynamic client fallback are converted to the same representation the param
// informer returns, so that policy evaluation behaves identically regardless of
// which path resolved the param.
func TestConvertParamToInformerRepresentation(t *testing.T) {
	tests := []struct {
		name          string
		gvk           schema.GroupVersionKind
		param         *unstructured.Unstructured
		wantObject    runtime.Object
		wantUnchanged bool
		wantErr       bool
	}{
		{
			name: "typed kind converts to typed object with empty TypeMeta",
			gvk:  schema.GroupVersionKind{Version: "v1", Kind: "ConfigMap"},
			param: &unstructured.Unstructured{
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
			},
			// Typed informers cache objects decoded by client-go, which have an empty TypeMeta.
			wantObject: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-param",
					Namespace: "default",
				},
				Data: map[string]string{
					"maxReplicas": "3",
				},
			},
		},
		{
			name: "unknown kind stays unstructured",
			gvk:  schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "TestParam"},
			param: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "example.com/v1",
					"kind":       "TestParam",
					"metadata": map[string]interface{}{
						"name":      "test-param",
						"namespace": "default",
					},
					"spec": map[string]interface{}{
						"maxReplicas": int64(3),
					},
				},
			},
			wantUnchanged: true,
		},
		{
			name: "typed kind with mismatched data errors",
			gvk:  schema.GroupVersionKind{Version: "v1", Kind: "ConfigMap"},
			param: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "ConfigMap",
					"metadata": map[string]interface{}{
						"name": "test-param",
					},
					"data": "not-a-map",
				},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			converted, err := convertParamToInformerRepresentation(tt.gvk, tt.param)
			if tt.wantErr {
				require.Error(t, err, "expected conversion error for malformed data")
				return
			}
			require.NoError(t, err, "unexpected error during conversion")
			if tt.wantUnchanged {
				require.Same(t, tt.param, converted, "expected the unstructured param to be returned unchanged")
				return
			}
			require.Equal(t, tt.wantObject, converted)
		})
	}
}

func TestCollectParamsWithContextReadsUnsyncedNamedParamDirectly(t *testing.T) {
	gvk := schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "TestParam"}
	gvr := gvk.GroupVersion().WithResource("testparams")
	mapping := &meta.RESTMapping{Resource: gvr, GroupVersionKind: gvk, Scope: meta.RESTScopeNamespace}
	param := newUnstructuredParam(gvk, "matching", "default", nil)
	client := dynamicfake.NewSimpleDynamicClient(runtime.NewScheme(), param)
	informer := dynamicinformer.NewFilteredDynamicInformer(
		client,
		gvr,
		metav1.NamespaceAll,
		0,
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
		nil,
	)
	require.False(t, informer.Informer().HasSynced())

	params, err := CollectParamsWithContext(
		context.Background(),
		&v1.ParamKind{APIVersion: gvk.GroupVersion().String(), Kind: gvk.Kind},
		informer,
		mapping,
		nil,
		&v1.ParamRef{Name: "matching", Namespace: "default"},
		"default",
		client,
	)

	require.NoError(t, err)
	require.Len(t, params, 1)
	accessor, err := meta.Accessor(params[0])
	require.NoError(t, err)
	require.Equal(t, "matching", accessor.GetName())
	require.Len(t, client.Actions(), 1)
	require.Equal(t, "get", client.Actions()[0].GetVerb())
}

func TestCollectParamsWithContextWaitsForUnsyncedSelector(t *testing.T) {
	gvk := schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "TestParam"}
	gvr := gvk.GroupVersion().WithResource("testparams")
	mapping := &meta.RESTMapping{Resource: gvr, GroupVersionKind: gvk, Scope: meta.RESTScopeNamespace}
	client := dynamicfake.NewSimpleDynamicClient(runtime.NewScheme())
	informer := dynamicinformer.NewFilteredDynamicInformer(
		client,
		gvr,
		metav1.NamespaceAll,
		0,
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
		nil,
	)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := CollectParamsWithContext(
		ctx,
		&v1.ParamKind{APIVersion: gvk.GroupVersion().String(), Kind: gvk.Kind},
		informer,
		mapping,
		nil,
		&v1.ParamRef{
			Namespace: "default",
			Selector:  &metav1.LabelSelector{MatchLabels: map[string]string{"selected": "true"}},
		},
		"default",
		client,
	)

	require.ErrorContains(t, err, "not yet synced to use for admission")
	require.Empty(t, client.Actions(), "selector references must not issue direct LIST requests")
}

func TestCollectParamsWithContextPropagatesCancellation(t *testing.T) {
	gvk := schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "TestParam"}
	gvr := gvk.GroupVersion().WithResource("testparams")
	mapping := &meta.RESTMapping{Resource: gvr, GroupVersionKind: gvk, Scope: meta.RESTScopeNamespace}
	client := dynamicfake.NewSimpleDynamicClient(runtime.NewScheme())
	informer := dynamicinformer.NewFilteredDynamicInformer(
		client,
		gvr,
		metav1.NamespaceAll,
		0,
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
		nil,
	)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := CollectParamsWithContext(
		ctx,
		&v1.ParamKind{APIVersion: gvk.GroupVersion().String(), Kind: gvk.Kind},
		informer,
		mapping,
		nil,
		&v1.ParamRef{Name: "test", Namespace: "default"},
		"default",
		contextAwareDynamicClient{Interface: client},
	)

	require.ErrorIs(t, err, context.Canceled)
}

func TestCollectParamsWithContextAppliesNotFoundActionToDirectReads(t *testing.T) {
	gvk := schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "TestParam"}
	gvr := gvk.GroupVersion().WithResource("testparams")
	mapping := &meta.RESTMapping{Resource: gvr, GroupVersionKind: gvk, Scope: meta.RESTScopeNamespace}

	tests := []struct {
		name      string
		paramRef  *v1.ParamRef
		wantError bool
	}{
		{
			name: "missing name is allowed",
			paramRef: &v1.ParamRef{
				Name:                    "missing",
				Namespace:               "default",
				ParameterNotFoundAction: ptr.To(v1.AllowAction),
			},
		},
		{
			name: "missing name is denied",
			paramRef: &v1.ParamRef{
				Name:                    "missing",
				Namespace:               "default",
				ParameterNotFoundAction: ptr.To(v1.DenyAction),
			},
			wantError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := dynamicfake.NewSimpleDynamicClientWithCustomListKinds(
				runtime.NewScheme(),
				map[schema.GroupVersionResource]string{gvr: "TestParamList"},
			)
			informer := dynamicinformer.NewFilteredDynamicInformer(
				client,
				gvr,
				metav1.NamespaceAll,
				0,
				cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
				nil,
			)

			params, err := CollectParamsWithContext(
				context.Background(),
				&v1.ParamKind{APIVersion: gvk.GroupVersion().String(), Kind: gvk.Kind},
				informer,
				mapping,
				nil,
				test.paramRef,
				"default",
				client,
			)

			if test.wantError {
				require.ErrorContains(t, err, "no params found")
			} else {
				require.NoError(t, err)
			}
			require.Empty(t, params)
		})
	}
}

func TestCollectParamsWithContextReturnsDirectReadErrors(t *testing.T) {
	gvk := schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "TestParam"}
	gvr := gvk.GroupVersion().WithResource("testparams")
	mapping := &meta.RESTMapping{Resource: gvr, GroupVersionKind: gvk, Scope: meta.RESTScopeNamespace}

	tests := []struct {
		name         string
		verb         string
		paramRef     *v1.ParamRef
		directErr    error
		wantInternal bool
	}{
		{
			name:      "name timeout",
			verb:      "get",
			paramRef:  &v1.ParamRef{Name: "test", Namespace: "default", ParameterNotFoundAction: ptr.To(v1.AllowAction)},
			directErr: context.DeadlineExceeded,
		},
		{
			name:         "name server error",
			verb:         "get",
			paramRef:     &v1.ParamRef{Name: "test", Namespace: "default", ParameterNotFoundAction: ptr.To(v1.AllowAction)},
			directErr:    apierrors.NewInternalError(errors.New("backend unavailable")),
			wantInternal: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := dynamicfake.NewSimpleDynamicClientWithCustomListKinds(
				runtime.NewScheme(),
				map[schema.GroupVersionResource]string{gvr: "TestParamList"},
			)
			client.PrependReactor(test.verb, gvr.Resource, func(clienttesting.Action) (bool, runtime.Object, error) {
				return true, nil, test.directErr
			})
			informer := dynamicinformer.NewFilteredDynamicInformer(
				client,
				gvr,
				metav1.NamespaceAll,
				0,
				cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
				nil,
			)

			params, err := CollectParamsWithContext(
				context.Background(),
				&v1.ParamKind{APIVersion: gvk.GroupVersion().String(), Kind: gvk.Kind},
				informer,
				mapping,
				nil,
				test.paramRef,
				"default",
				client,
			)

			require.Empty(t, params)
			if test.wantInternal {
				require.True(t, apierrors.IsInternalError(err), "expected internal error, got %v", err)
			} else {
				require.ErrorIs(t, err, context.DeadlineExceeded)
			}
			require.Len(t, client.Actions(), 1)
			require.Equal(t, test.verb, client.Actions()[0].GetVerb())
		})
	}
}

func newUnstructuredParam(gvk schema.GroupVersionKind, name, namespace string, labels map[string]string) *unstructured.Unstructured {
	unstructuredLabels := make(map[string]interface{}, len(labels))
	for key, value := range labels {
		unstructuredLabels[key] = value
	}
	param := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": gvk.GroupVersion().String(),
		"kind":       gvk.Kind,
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": namespace,
			"labels":    unstructuredLabels,
		},
	}}
	return param
}

type contextAwareDynamicClient struct {
	dynamic.Interface
}

func (c contextAwareDynamicClient) Resource(resource schema.GroupVersionResource) dynamic.NamespaceableResourceInterface {
	return &contextAwareNamespaceableResource{NamespaceableResourceInterface: c.Interface.Resource(resource)}
}

type contextAwareNamespaceableResource struct {
	dynamic.NamespaceableResourceInterface
}

func (r *contextAwareNamespaceableResource) Namespace(namespace string) dynamic.ResourceInterface {
	return &contextAwareResource{ResourceInterface: r.NamespaceableResourceInterface.Namespace(namespace)}
}

type contextAwareResource struct {
	dynamic.ResourceInterface
}

func (r *contextAwareResource) Get(ctx context.Context, name string, options metav1.GetOptions, subresources ...string) (*unstructured.Unstructured, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return r.ResourceInterface.Get(ctx, name, options, subresources...)
}
