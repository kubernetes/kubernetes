/*
Copyright 2016 The Kubernetes Authors.

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

package discovery

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	apidiscoveryv2beta1 "k8s.io/api/apidiscovery/v2beta1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensions "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	discoveryendpoint "k8s.io/apiserver/pkg/endpoints/discovery/aggregated"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	kubernetes "k8s.io/client-go/kubernetes"
	k8sscheme "k8s.io/client-go/kubernetes/scheme"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	apiregistrationv1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	aggregator "k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset"
	aggregatorclientsetscheme "k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset/scheme"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"

	"k8s.io/kubernetes/test/integration/framework"
)

type kubeClientSet = kubernetes.Interface
type aggegatorClientSet = aggregator.Interface
type apiextensionsClientSet = apiextensions.Interface
type dynamicClientset = dynamic.Interface
type testClientSet struct {
	kubeClientSet
	aggegatorClientSet
	apiextensionsClientSet
	dynamicClientset
}

func (t testClientSet) Discovery() discovery.DiscoveryInterface {
	return t.kubeClientSet.Discovery()
}

var (
	scheme    = runtime.NewScheme()
	codecs    = runtimeserializer.NewCodecFactory(scheme)
	serialize runtime.NegotiatedSerializer

	basicTestGroup = apidiscoveryv2beta1.APIGroupDiscovery{
		ObjectMeta: metav1.ObjectMeta{
			Name: "stable.example.com",
		},
		Versions: []apidiscoveryv2beta1.APIVersionDiscovery{
			{
				Version: "v1",
				Resources: []apidiscoveryv2beta1.APIResourceDiscovery{
					{
						Resource:   "jobs",
						Verbs:      []string{"create", "list", "watch", "delete"},
						ShortNames: []string{"jz"},
						Categories: []string{"all"},
					},
				},
				Freshness: apidiscoveryv2beta1.DiscoveryFreshnessCurrent,
			},
		},
	}
)

func init() {
	// Add all builtin types to scheme
	utilruntime.Must(k8sscheme.AddToScheme(scheme))
	utilruntime.Must(aggregatorclientsetscheme.AddToScheme(scheme))
	utilruntime.Must(apiextensionsv1.AddToScheme(scheme))

	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
	if !ok {
		panic("failed to create serializer info")
	}

	serialize = runtime.NewSimpleNegotiatedSerializer(info)
}

// Spins up an api server which is cleaned up at the end up the test
// Returns some kubernetes clients
func setup(t *testing.T) (context.Context, testClientSet, context.CancelFunc) {
	ctx, cancelCtx := context.WithCancel(context.Background())

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)

	kubeClientSet, err := kubernetes.NewForConfig(server.ClientConfig)
	require.NoError(t, err)

	aggegatorClientSet, err := aggregator.NewForConfig(server.ClientConfig)
	require.NoError(t, err)

	apiextensionsClientSet, err := apiextensions.NewForConfig(server.ClientConfig)
	require.NoError(t, err)

	dynamicClientset, err := dynamic.NewForConfig(server.ClientConfig)
	require.NoError(t, err)

	client := testClientSet{
		kubeClientSet:          kubeClientSet,
		aggegatorClientSet:     aggegatorClientSet,
		apiextensionsClientSet: apiextensionsClientSet,
		dynamicClientset:       dynamicClientset,
	}
	return ctx, client, cancelCtx
}

func registerAPIService(ctx context.Context, client aggregator.Interface, gv metav1.GroupVersion, service FakeService) error {
	port := service.Port()
	if port == nil {
		return errors.New("service not yet started")
	}
	// Register the APIService
	patch := apiregistrationv1.APIService{
		ObjectMeta: metav1.ObjectMeta{
			Name: gv.Version + "." + gv.Group,
		},
		TypeMeta: metav1.TypeMeta{
			Kind:       "APIService",
			APIVersion: "apiregistration.k8s.io/v1",
		},
		Spec: apiregistrationv1.APIServiceSpec{
			Group:                 gv.Group,
			Version:               gv.Version,
			InsecureSkipTLSVerify: true,
			GroupPriorityMinimum:  1000,
			VersionPriority:       15,
			Service: &apiregistrationv1.ServiceReference{
				Namespace: "default",
				Name:      service.Name(),
				Port:      port,
			},
		},
	}

	_, err := client.
		ApiregistrationV1().
		APIServices().
		Create(context.TODO(), &patch, metav1.CreateOptions{FieldManager: "test-manager"})
	return err
}

func unregisterAPIService(ctx context.Context, client aggregator.Interface, gv metav1.GroupVersion) error {
	return client.ApiregistrationV1().APIServices().Delete(ctx, gv.Version+"."+gv.Group, metav1.DeleteOptions{})
}

func WaitForGroupsAbsent(ctx context.Context, client testClientSet, groups ...string) error {
	return WaitForResultWithCondition(ctx, client, func(groupList apidiscoveryv2beta1.APIGroupDiscoveryList) bool {
		for _, searchGroup := range groups {
			for _, docGroup := range groupList.Items {
				if docGroup.Name == searchGroup {
					return false
				}
			}
		}
		return true
	})

}

func WaitForGroups(ctx context.Context, client testClientSet, groups ...apidiscoveryv2beta1.APIGroupDiscovery) error {
	return WaitForResultWithCondition(ctx, client, func(groupList apidiscoveryv2beta1.APIGroupDiscoveryList) bool {
		for _, searchGroup := range groups {
			for _, docGroup := range groupList.Items {
				if reflect.DeepEqual(searchGroup, docGroup) {
					return true
				}
			}
		}
		return false
	})
}

func WaitForResultWithCondition(ctx context.Context, client testClientSet, condition func(result apidiscoveryv2beta1.APIGroupDiscoveryList) bool) error {
	// Keep repeatedly fetching document from aggregator.
	// Check to see if it contains our service within a reasonable amount of time
	return wait.PollWithContext(
		ctx,
		250*time.Millisecond,
		1*time.Second,
		func(ctx context.Context) (done bool, err error) {
			result, err := client.
				Discovery().
				RESTClient().
				Get().
				AbsPath("/apis").
				SetHeader("Accept", "application/json;g=apidiscovery.k8s.io;v=v2beta1;as=APIGroupDiscoveryList").
				Do(ctx).
				Raw()

			if err != nil {
				return false, err
			}

			groupList := apidiscoveryv2beta1.APIGroupDiscoveryList{}
			err = json.Unmarshal(result, &groupList)
			if err != nil {
				panic(err)
			}

			if condition(groupList) {
				return true, nil
			}

			return false, nil
		})
}

func TestAggregatedAPIServiceDiscovery(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.AggregatedDiscoveryEndpoint, true)()

	// Keep any goroutines spawned from running past the execution of this test
	ctx, client, cleanup := setup(t)
	defer cleanup()

	// Create a resource manager whichs serves our GroupVersion
	resourceManager := discoveryendpoint.NewResourceManager()
	resourceManager.SetGroups([]apidiscoveryv2beta1.APIGroupDiscovery{basicTestGroup})

	// Install our ResourceManager as an Aggregated APIService to the
	// test server
	service := NewFakeService("test-server", client, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/apis") {
			resourceManager.ServeHTTP(w, r)
		} else if strings.HasPrefix(r.URL.Path, "/apis/stable.example.com") {
			// Return invalid response so APIService can be marked as "available"
			w.WriteHeader(http.StatusOK)
		} else {
			// reject openapi/v2, openapi/v3, apis/<group>/<version>
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	service.Start(t, ctx)

	// For each groupversion served by our resourcemanager, create an APIService
	// object connected to our fake APIServer
	for _, versionInfo := range basicTestGroup.Versions {
		groupVersion := metav1.GroupVersion{
			Group:   basicTestGroup.Name,
			Version: versionInfo.Version,
		}

		require.NoError(t, registerAPIService(ctx, client, groupVersion, service))
		defer func() {
			require.NoError(t, unregisterAPIService(ctx, client, groupVersion))
		}()
	}

	// Keep repeatedly fetching document from aggregator.
	// Check to see if it contains our service within a reasonable amount of time
	require.NoError(t, WaitForGroups(ctx, client, basicTestGroup))
}

// Shows that the following sequence is handled correctly:
// 1. Create an APIService
// - Check that API service is in discovery doc
// 2. Create CRD with the same GroupVersion as APIService
// 3. Delete APIService
// - Check that API service is removed from discovery
// 4. Update CRD
// -  Check that CRD is in discovery document
func TestOverlappingCRDAndAPIService(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.AggregatedDiscoveryEndpoint, true)()
	// Keep any goroutines spawned from running past the execution of this test
	ctx, client, cleanup := setup(t)
	defer cleanup()

	// Create a resource manager whichs serves our GroupVersion
	resourceManager := discoveryendpoint.NewResourceManager()
	resourceManager.SetGroups([]apidiscoveryv2beta1.APIGroupDiscovery{basicTestGroup})

	// Install our ResourceManager as an Aggregated APIService to the
	// test server
	service := NewFakeService("test-server", client, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/apis" {
			resourceManager.ServeHTTP(w, r)
		} else if strings.HasPrefix(r.URL.Path, "/apis/") {
			// Return "valid" response so APIService can be marked as "available"
			w.WriteHeader(http.StatusOK)
		} else {
			// reject openapi/v2, openapi/v3, apis/<group>/<version>
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	service.Start(t, ctx)

	// For each groupversion served by our resourcemanager, create an APIService
	// object connected to our fake APIServer
	for _, versionInfo := range basicTestGroup.Versions {
		groupVersion := metav1.GroupVersion{
			Group:   basicTestGroup.Name,
			Version: versionInfo.Version,
		}

		registerAPIService(ctx, client, groupVersion, service)
	}

	// Keep repeatedly fetching document from aggregator.
	// Check to see if it contains our service within a reasonable amount of time
	require.NoError(t, WaitForGroups(ctx, client, basicTestGroup))

	// Create a CRD
	crd, err := client.ApiextensionsV1().CustomResourceDefinitions().Create(ctx, &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foos.stable.example.com",
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "stable.example.com",
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Singular: "foo",
				Plural:   "foos",
				Kind:     "Foo",
			},
			Scope: apiextensionsv1.ClusterScoped,
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{
					Name:    "v1",
					Served:  true,
					Storage: true,
					Schema: &apiextensionsv1.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensionsv1.JSONSchemaProps{
								"stringMap": {
									Description: "a map[string]string",
									Type:        "object",
									AdditionalProperties: &apiextensionsv1.JSONSchemaPropsOrBool{
										Schema: &apiextensionsv1.JSONSchemaProps{
											Type: "string",
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}, metav1.CreateOptions{
		FieldManager: "test-manager",
	})
	require.NoError(t, err)

	// Create a CR for the CRD
	// Keep trying until it succeeds (or should we try for discovery?)
	require.NoError(t, wait.PollWithContext(ctx, 100*time.Millisecond, 1*time.Second, func(ctx context.Context) (done bool, err error) {
		toCreate := &unstructured.Unstructured{}
		toCreate.SetUnstructuredContent(map[string]any{
			"apiVersion": "stable.example.com/v1",
			"kind":       "Foo",
			"key":        "value",
		})

		_, err = client.dynamicClientset.Resource(schema.GroupVersionResource{
			Group:    "stable.example.com",
			Version:  "v1",
			Resource: "foos",
		}).Create(ctx, toCreate, metav1.CreateOptions{
			FieldManager: "test-manager",
		})
		return err != nil, nil
	}))

	// For each groupversion served by our resourcemanager, delete an APIService
	// object connected to our fake APIServer
	for _, versionInfo := range basicTestGroup.Versions {
		groupVersion := metav1.GroupVersion{
			Group:   basicTestGroup.Name,
			Version: versionInfo.Version,
		}

		unregisterAPIService(ctx, client, groupVersion)
	}

	// Wait for the apiservice to be deleted from discovery
	require.NoError(t, WaitForGroupsAbsent(ctx, client, "stable.example.com"))

	// Update the CRD with a minor change to show that reconciliation will
	// eventually refresh the discovery group on resync
	obj := &unstructured.Unstructured{}
	obj.SetUnstructuredContent(map[string]interface{}{
		"apiVersion": "apiextensions.k8s.io/v1",
		"kind":       "CustomResourceDefinition",
		"metadata": map[string]any{
			"name": crd.Name,
		},
		"spec": map[string]interface{}{
			"names": map[string]any{
				"categories": []string{"all"},
			},
		},
	})

	buf := bytes.NewBuffer(nil)
	err = unstructured.UnstructuredJSONScheme.Encode(obj, buf)
	require.NoError(t, err)

	//Is there a better way to force crd resync?
	_, err = client.ApiextensionsV1().CustomResourceDefinitions().Patch(
		ctx,
		crd.Name,
		types.ApplyPatchType,
		buf.Bytes(),
		metav1.PatchOptions{
			FieldManager: "test-manager",
		},
	)
	require.NoError(t, err)

	// Wait until the crd appears in discovery
	expectedDiscovery := apidiscoveryv2beta1.APIGroupDiscovery{
		ObjectMeta: metav1.ObjectMeta{
			Name: basicTestGroup.Name,
		},
		Versions: []apidiscoveryv2beta1.APIVersionDiscovery{
			{
				Version: "v1",
				Resources: []apidiscoveryv2beta1.APIResourceDiscovery{
					{
						Resource: "foos",
						ResponseKind: &metav1.GroupVersionKind{
							Group:   basicTestGroup.Name,
							Version: "v1",
							Kind:    "Foo",
						},
						Scope:            apidiscoveryv2beta1.ScopeCluster,
						SingularResource: crd.Spec.Names.Singular,
						Verbs:            []string{"delete", "deletecollection", "get", "list", "patch", "create", "update", "watch"},
						Categories:       []string{"all"},
					},
				},
				//!TODO: set freshness of builtin/crds
				Freshness: "",
			},
		},
	}
	require.NoError(t, WaitForGroups(ctx, client, expectedDiscovery))
}
