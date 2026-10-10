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

package admission

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsfake "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset/fake"
	apiextensionsinformers "k8s.io/apiextensions-apiserver/pkg/client/informers/externalversions"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	kubetesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
)

func TestCRDParamKindResolver(t *testing.T) {
	paramKind := schema.GroupVersionKind{Group: "params.example.com", Version: "v1", Kind: "ExampleParam"}
	tests := []struct {
		name        string
		crd         *apiextensionsv1.CustomResourceDefinition
		want        *meta.RESTMapping
		wantHandles bool
	}{
		{
			name: "resolves established served CRD",
			crd:  establishedCRD(paramKind, true, apiextensionsv1.NamespaceScoped),
			want: &meta.RESTMapping{
				Resource:         paramKind.GroupVersion().WithResource("exampleparams"),
				GroupVersionKind: paramKind,
				Scope:            meta.RESTScopeNamespace,
			},
			wantHandles: true,
		},
		{
			name: "resolves cluster scoped CRD",
			crd:  establishedCRD(paramKind, true, apiextensionsv1.ClusterScoped),
			want: &meta.RESTMapping{
				Resource:         paramKind.GroupVersion().WithResource("exampleparams"),
				GroupVersionKind: paramKind,
				Scope:            meta.RESTScopeRoot,
			},
			wantHandles: true,
		},
		{
			name: "ignores unserved version",
			crd:  establishedCRD(paramKind, false, apiextensionsv1.NamespaceScoped),
		},
		{
			name: "ignores unestablished CRD",
			crd: func() *apiextensionsv1.CustomResourceDefinition {
				crd := establishedCRD(paramKind, true, apiextensionsv1.NamespaceScoped)
				crd.Status.Conditions = nil
				return crd
			}(),
		},
		{
			name: "ignores absent CRD",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var objects []runtime.Object
			if test.crd != nil {
				objects = append(objects, test.crd)
			}
			resolver, _ := startTestCRDParamKindResolver(t, objects...)

			mapping, err := resolver.Resolve(context.Background(), paramKind)

			require.NoError(t, err)
			require.Equal(t, test.want, mapping)
			require.Equal(t, test.wantHandles, resolver.Handles(paramKind))
		})
	}
}

func TestCRDParamKindResolverClaimsOnlyServedAcceptedVersions(t *testing.T) {
	acceptedParamKind := schema.GroupVersionKind{Group: "params.example.com", Version: "v2", Kind: "ExampleParam"}
	crd := establishedCRD(acceptedParamKind, true, apiextensionsv1.NamespaceScoped)
	crd.Spec.Names.Kind = "ProposedParam"
	crd.Spec.Versions = append(crd.Spec.Versions, apiextensionsv1.CustomResourceDefinitionVersion{Name: "v1", Served: false})
	resolver, _ := startTestCRDParamKindResolver(t, crd)

	require.True(t, resolver.Handles(acceptedParamKind))
	require.False(t, resolver.Handles(schema.GroupVersionKind{Group: acceptedParamKind.Group, Version: "v1", Kind: acceptedParamKind.Kind}))
	require.False(t, resolver.Handles(schema.GroupVersionKind{Group: acceptedParamKind.Group, Version: acceptedParamKind.Version, Kind: crd.Spec.Names.Kind}))
}

func TestCRDParamKindResolverRemainsAuthoritativeAfterVersionUnserved(t *testing.T) {
	paramKind := schema.GroupVersionKind{Group: "params.example.com", Version: "v1", Kind: "ExampleParam"}
	crd := establishedCRD(paramKind, true, apiextensionsv1.NamespaceScoped)
	resolver, client := startTestCRDParamKindResolver(t, crd)
	changes := make(chan schema.GroupKind, 1)
	unregister := resolver.RegisterForChanges(func(groupKind schema.GroupKind) {
		changes <- groupKind
	})
	t.Cleanup(unregister)

	updated := crd.DeepCopy()
	updated.Spec.Versions[0].Served = false
	_, err := client.ApiextensionsV1().CustomResourceDefinitions().Update(context.Background(), updated, metav1.UpdateOptions{})
	require.NoError(t, err)
	select {
	case groupKind := <-changes:
		require.Equal(t, paramKind.GroupKind(), groupKind)
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for CRD update notification")
	}

	mapping, err := resolver.Resolve(context.Background(), paramKind)
	require.NoError(t, err)
	require.Nil(t, mapping)
	require.True(t, resolver.Handles(paramKind))
}

func TestCRDParamKindResolverRemainsAuthoritativeAfterDeletion(t *testing.T) {
	paramKind := schema.GroupVersionKind{Group: "params.example.com", Version: "v1", Kind: "ExampleParam"}
	crd := establishedCRD(paramKind, true, apiextensionsv1.NamespaceScoped)
	resolver, client := startTestCRDParamKindResolver(t, crd)
	changes := make(chan schema.GroupKind, 1)
	unregister := resolver.RegisterForChanges(func(groupKind schema.GroupKind) {
		changes <- groupKind
	})
	t.Cleanup(unregister)

	require.NoError(t, client.ApiextensionsV1().CustomResourceDefinitions().Delete(context.Background(), crd.Name, metav1.DeleteOptions{}))
	select {
	case groupKind := <-changes:
		require.Equal(t, paramKind.GroupKind(), groupKind)
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for CRD deletion notification")
	}

	mapping, err := resolver.Resolve(context.Background(), paramKind)
	require.NoError(t, err)
	require.Nil(t, mapping)
	require.True(t, resolver.Handles(paramKind))
}

func TestCRDParamKindResolverDoesNotListDuringResolution(t *testing.T) {
	paramKind := schema.GroupVersionKind{Group: "params.example.com", Version: "v1", Kind: "ExampleParam"}
	client := apiextensionsfake.NewSimpleClientset(establishedCRD(paramKind, true, apiextensionsv1.NamespaceScoped))
	var listCalls atomic.Int32
	client.PrependReactor("list", "customresourcedefinitions", func(kubetesting.Action) (bool, runtime.Object, error) {
		listCalls.Add(1)
		return false, nil, nil
	})
	resolver := startTestCRDParamKindResolverWithClient(t, client)

	_, err := resolver.Resolve(context.Background(), paramKind)
	require.NoError(t, err)
	_, err = resolver.Resolve(context.Background(), paramKind)
	require.NoError(t, err)

	require.EqualValues(t, 1, listCalls.Load())
}

func TestCRDParamKindResolverSyncExpectation(t *testing.T) {
	resolver := newCRDParamKindResolver()
	require.True(t, resolver.HasSynced())

	resolver.ExpectCustomResourceDefinitionInformer()
	require.False(t, resolver.HasSynced())
}

func TestCRDParamKindResolverWaitsForHandlerSync(t *testing.T) {
	paramKind := schema.GroupVersionKind{Group: "params.example.com", Version: "v1", Kind: "ExampleParam"}
	client := apiextensionsfake.NewSimpleClientset(establishedCRD(paramKind, true, apiextensionsv1.NamespaceScoped))
	factory := apiextensionsinformers.NewSharedInformerFactory(client, 0)
	informer := factory.Apiextensions().V1().CustomResourceDefinitions()
	resolver := newCRDParamKindResolver()
	require.NoError(t, resolver.SetCustomResourceDefinitionInformer(informer))

	handlerStarted := make(chan struct{})
	releaseHandler := make(chan struct{})
	resolver.RegisterForChanges(func(schema.GroupKind) {
		close(handlerStarted)
		<-releaseHandler
	})
	t.Cleanup(func() {
		select {
		case <-releaseHandler:
		default:
			close(releaseHandler)
		}
	})

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	factory.Start(ctx.Done())
	require.True(t, cache.WaitForCacheSync(ctx.Done(), informer.Informer().HasSynced))
	select {
	case <-handlerStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for initial CRD event")
	}
	require.False(t, resolver.HasSynced())

	close(releaseHandler)
	require.True(t, cache.WaitForCacheSync(ctx.Done(), resolver.HasSynced))
}

func TestCRDParamKindResolverNotifiesOnCRDChanges(t *testing.T) {
	paramKind := schema.GroupVersionKind{Group: "params.example.com", Version: "v1", Kind: "ExampleParam"}
	resolver, client := startTestCRDParamKindResolver(t)
	changes := make(chan schema.GroupKind, 1)
	unregister := resolver.RegisterForChanges(func(groupKind schema.GroupKind) {
		changes <- groupKind
	})
	t.Cleanup(unregister)

	_, err := client.ApiextensionsV1().CustomResourceDefinitions().Create(
		context.Background(),
		establishedCRD(paramKind, true, apiextensionsv1.NamespaceScoped),
		metav1.CreateOptions{},
	)
	require.NoError(t, err)

	select {
	case groupKind := <-changes:
		require.Equal(t, paramKind.GroupKind(), groupKind)
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for CRD change notification")
	}
}

func startTestCRDParamKindResolver(t *testing.T, objects ...runtime.Object) (*crdParamKindResolver, *apiextensionsfake.Clientset) {
	t.Helper()
	client := apiextensionsfake.NewSimpleClientset(objects...)
	return startTestCRDParamKindResolverWithClient(t, client), client
}

func startTestCRDParamKindResolverWithClient(t *testing.T, client *apiextensionsfake.Clientset) *crdParamKindResolver {
	t.Helper()
	factory := apiextensionsinformers.NewSharedInformerFactory(client, 0)
	informer := factory.Apiextensions().V1().CustomResourceDefinitions()
	resolver := newCRDParamKindResolver()
	require.NoError(t, resolver.SetCustomResourceDefinitionInformer(informer))
	require.False(t, resolver.HasSynced())

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	factory.Start(ctx.Done())
	require.True(t, cache.WaitForCacheSync(ctx.Done(), informer.Informer().HasSynced))
	require.True(t, resolver.HasSynced())
	return resolver
}

func establishedCRD(paramKind schema.GroupVersionKind, served bool, scope apiextensionsv1.ResourceScope) *apiextensionsv1.CustomResourceDefinition {
	return &apiextensionsv1.CustomResourceDefinition{
		TypeMeta: metav1.TypeMeta{
			APIVersion: apiextensionsv1.SchemeGroupVersion.String(),
			Kind:       "CustomResourceDefinition",
		},
		ObjectMeta: metav1.ObjectMeta{Name: "exampleparams." + paramKind.Group},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: paramKind.Group,
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural: "exampleparams",
				Kind:   paramKind.Kind,
			},
			Scope: scope,
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{Name: paramKind.Version, Served: served, Storage: true},
			},
		},
		Status: apiextensionsv1.CustomResourceDefinitionStatus{
			AcceptedNames: apiextensionsv1.CustomResourceDefinitionNames{
				Plural: "exampleparams",
				Kind:   paramKind.Kind,
			},
			Conditions: []apiextensionsv1.CustomResourceDefinitionCondition{
				{Type: apiextensionsv1.Established, Status: apiextensionsv1.ConditionTrue},
			},
		},
	}
}
