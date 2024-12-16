/*
Copyright 2017 The Kubernetes Authors.

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

package fixtures

import (
	"context"
	"fmt"
	"time"

	"k8s.io/utils/pointer"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/scale"
)

const (
	noxuInstanceNum int64 = 9223372036854775807
)

// AllowAllSchema doesn't enforce any schema restrictions
func AllowAllSchema() *apiextensionsv1.CustomResourceValidation {
	return &apiextensionsv1.CustomResourceValidation{
		OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
			XPreserveUnknownFields: pointer.BoolPtr(true),
			Type:                   "object",
		},
	}
}

// NewRandomNameV1CustomResourceDefinition generates a CRD with random name to avoid name conflict in e2e tests
func NewRandomNameV1CustomResourceDefinition(scope apiextensionsv1.ResourceScope) *apiextensionsv1.CustomResourceDefinition {
	// ensure the singular doesn't end in an s for now
	gName := names.SimpleNameGenerator.GenerateName("foo") + "a"
	return &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: gName + "s.mygroup.example.com"},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "mygroup.example.com",
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{
					Name:    "v1beta1",
					Served:  true,
					Storage: true,
					Schema:  AllowAllSchema(),
				},
			},
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural:   gName + "s",
				Singular: gName,
				Kind:     gName,
				ListKind: gName + "List",
			},
			Scope: scope,
		},
	}
}

// NewRandomNameMultipleCustomResourceDefinition generates a multi version CRD with random name to avoid name conflict in e2e tests
func NewRandomNameMultipleVersionCustomResourceDefinition(scope apiextensionsv1.ResourceScope) *apiextensionsv1.CustomResourceDefinition {
	// ensure the singular doesn't end in an s for now
	gName := names.SimpleNameGenerator.GenerateName("foo") + "a"
	return &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: gName + "s.mygroup.example.com"},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "mygroup.example.com",
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{
					Name:    "v1beta1",
					Served:  true,
					Storage: false,
					Subresources: &apiextensionsv1.CustomResourceSubresources{
						Status: &apiextensionsv1.CustomResourceSubresourceStatus{},
					},
					Schema: AllowAllSchema(),
				},
				{
					Name:    "v1",
					Served:  true,
					Storage: true,
					Subresources: &apiextensionsv1.CustomResourceSubresources{
						Status: &apiextensionsv1.CustomResourceSubresourceStatus{},
					},
					Schema: AllowAllSchema(),
				},
			},
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural:   gName + "s",
				Singular: gName,
				Kind:     gName,
				ListKind: gName + "List",
			},
			Scope: scope,
		},
	}
}

// NewRandomNameV1CustomResourceDefinitionWithSchema generates a CRD with random name and the provided OpenAPIv3 schema to avoid name conflict in e2e tests
func NewRandomNameV1CustomResourceDefinitionWithSchema(scope apiextensionsv1.ResourceScope, openAPIV3Schema *apiextensionsv1.JSONSchemaProps, enableStatus bool) *apiextensionsv1.CustomResourceDefinition {
	crd := NewRandomNameV1CustomResourceDefinition(scope)
	for i := range crd.Spec.Versions {
		if enableStatus {
			crd.Spec.Versions[i].Subresources = &apiextensionsv1.CustomResourceSubresources{
				Status: &apiextensionsv1.CustomResourceSubresourceStatus{},
			}
		}
		crd.Spec.Versions[i].Schema = &apiextensionsv1.CustomResourceValidation{OpenAPIV3Schema: openAPIV3Schema}
	}
	return crd
}

// GetGroupVersionResourcesOfCustomResource gets all GroupVersionResources for custom resources of the CustomResourceDefinition.
func GetGroupVersionResourcesOfCustomResource(crd *apiextensionsv1.CustomResourceDefinition) []schema.GroupVersionResource {
	var result []schema.GroupVersionResource
	for _, v := range crd.Spec.Versions {
		result = append(result, schema.GroupVersionResource{
			Group:    crd.Spec.Group,
			Version:  v.Name,
			Resource: crd.Spec.Names.Plural,
		})
	}
	return result
}

// NewNoxuV1CustomResourceDefinition returns a WishIHadChosenNoxu CRD.
func NewNoxuV1CustomResourceDefinition(scope apiextensionsv1.ResourceScope) *apiextensionsv1.CustomResourceDefinition {
	return &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: "noxus.mygroup.example.com"},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "mygroup.example.com",
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{{
				Name:    "v1beta1",
				Served:  true,
				Storage: true,
				Schema:  AllowAllSchema(),
			}},
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural:     "noxus",
				Singular:   "nonenglishnoxu",
				Kind:       "WishIHadChosenNoxu",
				ShortNames: []string{"foo", "bar", "abc", "def"},
				ListKind:   "NoxuItemList",
				Categories: []string{"all"},
			},
			Scope: scope,
		},
	}
}

// NewVersionedNoxuInstance returns a WishIHadChosenNoxu instance for a given version
func NewVersionedNoxuInstance(namespace, name, version string) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "mygroup.example.com/" + version,
			"kind":       "WishIHadChosenNoxu",
			"metadata": map[string]interface{}{
				"namespace": namespace,
				"name":      name,
			},
			"content": map[string]interface{}{
				"key": "value",
			},
			"num": map[string]interface{}{
				"num1": noxuInstanceNum,
				"num2": 1000000,
			},
		},
	}
}

// NewNoxuInstance returns a WishIHadChosenNoxu instance for v1beta1.
func NewNoxuInstance(namespace, name string) *unstructured.Unstructured {
	return NewVersionedNoxuInstance(namespace, name, "v1beta1")
}

// NewMultipleVersionNoxuCRD returns a WishIHadChosenNoxu with multiple versions.
func NewMultipleVersionNoxuCRD(scope apiextensionsv1.ResourceScope) *apiextensionsv1.CustomResourceDefinition {
	return &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: "noxus.mygroup.example.com"},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "mygroup.example.com",
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural:     "noxus",
				Singular:   "nonenglishnoxu",
				Kind:       "WishIHadChosenNoxu",
				ShortNames: []string{"foo", "bar", "abc", "def"},
				ListKind:   "NoxuItemList",
				Categories: []string{"all"},
			},
			Scope: scope,
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{
					Name:    "v1beta1",
					Served:  true,
					Storage: false,
					Subresources: &apiextensionsv1.CustomResourceSubresources{
						Status: &apiextensionsv1.CustomResourceSubresourceStatus{},
					},
					Schema: AllowAllSchema(),
				},
				{
					Name:    "v1beta2",
					Served:  true,
					Storage: true,
					Subresources: &apiextensionsv1.CustomResourceSubresources{
						Status: &apiextensionsv1.CustomResourceSubresourceStatus{},
					},
					Schema: AllowAllSchema(),
				},
				{
					Name:    "v0",
					Served:  false,
					Storage: false,
					Subresources: &apiextensionsv1.CustomResourceSubresources{
						Status: &apiextensionsv1.CustomResourceSubresourceStatus{},
					},
					Schema: AllowAllSchema(),
				},
			},
		},
	}
}

// NewNoxu2CustomResourceDefinition returns a WishIHadChosenNoxu2 CRD.
func NewNoxu2CustomResourceDefinition(scope apiextensionsv1.ResourceScope) *apiextensionsv1.CustomResourceDefinition {
	return &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: "noxus2.mygroup.example.com"},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "mygroup.example.com",
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{{
				Name:    "v1alpha1",
				Served:  true,
				Storage: true,
				Schema:  AllowAllSchema(),
			}},
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural:     "noxus2",
				Singular:   "nonenglishnoxu2",
				Kind:       "WishIHadChosenNoxu2",
				ShortNames: []string{"foo", "bar", "abc", "def"},
				ListKind:   "Noxu2ItemList",
			},
			Scope: scope,
		},
	}
}

// NewCurletV1CustomResourceDefinition returns a Curlet CRD.
func NewCurletV1CustomResourceDefinition(scope apiextensionsv1.ResourceScope) *apiextensionsv1.CustomResourceDefinition {
	return &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: "curlets.mygroup.example.com"},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "mygroup.example.com",
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{
					Name:    "v1beta1",
					Served:  true,
					Storage: true,
					Schema:  AllowAllSchema(),
				},
			},
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural:   "curlets",
				Singular: "curlet",
				Kind:     "Curlet",
				ListKind: "CurletList",
			},
			Scope: scope,
		},
	}
}

// NewCurletInstance returns a Curlet instance.
func NewCurletInstance(namespace, name string) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "mygroup.example.com/v1beta1",
			"kind":       "Curlet",
			"metadata": map[string]interface{}{
				"namespace": namespace,
				"name":      name,
			},
			"content": map[string]interface{}{
				"key": "value",
			},
		},
	}
}

func servedV1Versions(crd *apiextensionsv1.CustomResourceDefinition) []string {
	if len(crd.Spec.Versions) == 0 {
		return []string{}
	}
	var versions []string
	for _, v := range crd.Spec.Versions {
		if v.Served {
			versions = append(versions, v.Name)
		}
	}
	return versions
}

func existsInDiscoveryV1(crd *apiextensionsv1.CustomResourceDefinition, apiExtensionsClient clientset.Interface, version string) (bool, error) {
	groupResource, err := apiExtensionsClient.Discovery().ServerResourcesForGroupVersion(crd.Spec.Group + "/" + version)
	if err != nil {
		if errors.IsNotFound(err) {
			return false, nil
		}
		return false, err
	}
	for _, g := range groupResource.APIResources {
		if g.Name == crd.Spec.Names.Plural {
			return true, nil
		}
	}
	return false, nil
}

// waitForCRDReadyWatchUnsafe creates the CRD and makes sure
// the apiextension apiserver has installed the CRD. But it's not safe to watch
// the created CR. Please call CreateCRDUsingRemovedAPI if you need to
// watch the CR.
func waitForCRDReadyWatchUnsafe(crd *apiextensionsv1.CustomResourceDefinition, apiExtensionsClient clientset.Interface) (*apiextensionsv1.CustomResourceDefinition, error) {
	// wait until all resources appears in discovery
	for _, version := range servedV1Versions(crd) {
		err := wait.PollUntilContextTimeout(context.Background(), 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			return existsInDiscoveryV1(crd, apiExtensionsClient, version)
		})
		if err != nil {
			return nil, err
		}
	}

	return crd, nil
}

// waitForCRDReady creates the given CRD and makes sure its watch cache is primed on the server.
func waitForCRDReady(crd *apiextensionsv1.CustomResourceDefinition, apiExtensionsClient clientset.Interface, dynamicClientSet dynamic.Interface) (*apiextensionsv1.CustomResourceDefinition, error) {
	v1CRD, err := waitForCRDReadyWatchUnsafe(crd, apiExtensionsClient)
	if err != nil {
		return nil, err
	}

	// This is only for a test.  We need the watch cache to have a resource version that works for the test.
	// When new REST storage is created, the storage cacher for the CR starts asynchronously.
	// REST API operations return like list use the RV of etcd, but the storage cacher's reflector's list
	// can get a different RV because etcd can be touched in between the initial list operation (if that's what you're doing first)
	// and the storage cache reflector starting.
	// Later, you can issue a watch with the REST apis list.RV and end up earlier than the storage cacher.
	// The general working model is that if you get a "resourceVersion too old" message, you re-list and rewatch.
	// For this test, we'll actually cycle, "list/watch/create/delete" until we get an RV from list that observes the create and not an error.
	// This way all the tests that are checking for watches don't have to worry about RV too old problems because crazy things *could* happen
	// before like the created RV could be too old to watch.
	err = wait.PollUntilContextTimeout(context.Background(), 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		return isWatchCachePrimed(v1CRD, dynamicClientSet)
	})
	if err != nil {
		return nil, err
	}
	return v1CRD, nil
}

// CreateNewV1CustomResourceDefinitionWatchUnsafe creates the CRD and makes sure
// the apiextension apiserver has installed the CRD. But it's not safe to watch
// the created CR. Please call CreateNewV1CustomResourceDefinition if you need to
// watch the CR.
func CreateNewV1CustomResourceDefinitionWatchUnsafe(v1CRD *apiextensionsv1.CustomResourceDefinition, apiExtensionsClient clientset.Interface) (*apiextensionsv1.CustomResourceDefinition, error) {
	v1CRD, err := apiExtensionsClient.ApiextensionsV1().CustomResourceDefinitions().Create(context.TODO(), v1CRD, metav1.CreateOptions{})
	if err != nil {
		return nil, err
	}

	// wait until all resources appears in discovery
	for _, version := range servedV1Versions(v1CRD) {
		err := wait.PollUntilContextTimeout(context.Background(), 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			return existsInDiscoveryV1(v1CRD, apiExtensionsClient, version)
		})
		if err != nil {
			return nil, err
		}
	}

	return v1CRD, nil
}

// CreateNewV1CustomResourceDefinition creates the given CRD and makes sure its watch cache is primed on the server.
func CreateNewV1CustomResourceDefinition(v1CRD *apiextensionsv1.CustomResourceDefinition, apiExtensionsClient clientset.Interface, dynamicClientSet dynamic.Interface) (*apiextensionsv1.CustomResourceDefinition, error) {
	v1CRD, err := CreateNewV1CustomResourceDefinitionWatchUnsafe(v1CRD, apiExtensionsClient)
	if err != nil {
		return nil, err
	}

	// This is only for a test.  We need the watch cache to have a resource version that works for the test.
	// When new REST storage is created, the storage cacher for the CR starts asynchronously.
	// REST API operations return like list use the RV of etcd, but the storage cacher's reflector's list
	// can get a different RV because etcd can be touched in between the initial list operation (if that's what you're doing first)
	// and the storage cache reflector starting.
	// Later, you can issue a watch with the REST apis list.RV and end up earlier than the storage cacher.
	// The general working model is that if you get a "resourceVersion too old" message, you re-list and rewatch.
	// For this test, we'll actually cycle, "list/watch/create/delete" until we get an RV from list that observes the create and not an error.
	// This way all the tests that are checking for watches don't have to worry about RV too old problems because crazy things *could* happen
	// before like the created RV could be too old to watch.
	err = wait.PollUntilContextTimeout(context.Background(), 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		return isWatchCachePrimed(v1CRD, dynamicClientSet)
	})
	if err != nil {
		return nil, err
	}
	return v1CRD, nil
}

func resourceClientForVersion(crd *apiextensionsv1.CustomResourceDefinition, dynamicClientSet dynamic.Interface, namespace, version string) dynamic.ResourceInterface {
	gvr := schema.GroupVersionResource{Group: crd.Spec.Group, Version: version, Resource: crd.Spec.Names.Plural}
	if crd.Spec.Scope != apiextensionsv1.ClusterScoped {
		return dynamicClientSet.Resource(gvr).Namespace(namespace)
	}
	return dynamicClientSet.Resource(gvr)
}

// isWatchCachePrimed returns true if the watch is primed for an specified version of CRD watch
func isWatchCachePrimed(crd *apiextensionsv1.CustomResourceDefinition, dynamicClientSet dynamic.Interface) (bool, error) {
	ns := ""
	if crd.Spec.Scope != apiextensionsv1.ClusterScoped {
		ns = "default"
	}

	versions := servedV1Versions(crd)
	if len(versions) == 0 {
		return true, nil
	}

	resourceClient := resourceClientForVersion(crd, dynamicClientSet, ns, versions[0])
	instanceName := "setup-instance"
	instance := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": crd.Spec.Group + "/" + versions[0],
			"kind":       crd.Spec.Names.Kind,
			"metadata": map[string]interface{}{
				"namespace": ns,
				"name":      instanceName,
			},
			"alpha":   "foo_123",
			"beta":    10,
			"gamma":   "bar",
			"delta":   "hello",
			"epsilon": "foobar",
			"spec":    map[string]interface{}{},
		},
	}
	createdInstance, err := resourceClient.Create(context.TODO(), instance, metav1.CreateOptions{})
	if err != nil {
		return false, err
	}
	err = resourceClient.Delete(context.TODO(), createdInstance.GetName(), metav1.DeleteOptions{})
	if err != nil {
		return false, err
	}

	// Wait for all versions of watch cache to be primed and also make sure we consumed the DELETE event for all
	// versions so that any new watch with ResourceVersion=0 does not get those events. This is source of some flaky tests.
	// When a client creates a watch with resourceVersion=0, it will get an ADD event for any existing objects
	// but because they specified resourceVersion=0, there is no starting point in the cache buffer to return existing events
	// from, thus the server will return anything from current head of the cache to the end. By accessing the delete
	// events for all versions here, we make sure that the head of the cache is passed those events and they will not being
	// delivered to any future watch with resourceVersion=0.
	for _, v := range versions {
		noxuWatch, err := resourceClientForVersion(crd, dynamicClientSet, ns, v).Watch(
			context.TODO(),
			metav1.ListOptions{ResourceVersion: createdInstance.GetResourceVersion()})
		if err != nil {
			return false, err
		}
		defer noxuWatch.Stop()

		select {
		case watchEvent := <-noxuWatch.ResultChan():
			if watch.Error == watchEvent.Type {
				return false, nil
			}
			if watch.Deleted != watchEvent.Type {
				return false, fmt.Errorf("expected DELETE, but got %#v", watchEvent)
			}
		case <-time.After(5 * time.Second):
			return false, fmt.Errorf("gave up waiting for watch event")
		}
	}

	return true, nil
}

// DeleteV1CustomResourceDefinition deletes a CRD and waits until it disappears from discovery.
func DeleteV1CustomResourceDefinition(crd *apiextensionsv1.CustomResourceDefinition, apiExtensionsClient clientset.Interface) error {
	if err := apiExtensionsClient.ApiextensionsV1().CustomResourceDefinitions().Delete(context.TODO(), crd.Name, metav1.DeleteOptions{}); err != nil {
		return err
	}
	for _, version := range servedV1Versions(crd) {
		err := wait.PollUntilContextTimeout(context.Background(), 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			exists, err := existsInDiscoveryV1(crd, apiExtensionsClient, version)
			return !exists, err
		})
		if err != nil {
			return err
		}
	}
	return nil
}

// DeleteV1CustomResourceDefinitions deletes all CRD matching the provided deleteListOpts and waits until all the CRDs disappear from discovery.
func DeleteV1CustomResourceDefinitions(deleteListOpts metav1.ListOptions, apiExtensionsClient clientset.Interface) error {
	list, err := apiExtensionsClient.ApiextensionsV1().CustomResourceDefinitions().List(context.TODO(), deleteListOpts)
	if err != nil {
		return err
	}
	if err = apiExtensionsClient.ApiextensionsV1().CustomResourceDefinitions().DeleteCollection(context.TODO(), metav1.DeleteOptions{}, deleteListOpts); err != nil {
		return err
	}
	for _, crd := range list.Items {
		for _, version := range servedV1Versions(&crd) {
			err := wait.PollUntilContextTimeout(context.Background(), 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
				exists, err := existsInDiscoveryV1(&crd, apiExtensionsClient, version)
				return !exists, err
			})
			if err != nil {
				return err
			}
		}
	}
	return nil
}

// CreateNewVersionedScaleClient returns a scale client.
func CreateNewVersionedScaleClient(crd *apiextensionsv1.CustomResourceDefinition, config *rest.Config, version string) (scale.ScalesGetter, error) {
	discoveryClient, err := discovery.NewDiscoveryClientForConfig(config)
	if err != nil {
		return nil, err
	}
	groupResource, err := discoveryClient.ServerResourcesForGroupVersion(crd.Spec.Group + "/" + version)
	if err != nil {
		return nil, err
	}

	resources := []*restmapper.APIGroupResources{
		{
			Group: metav1.APIGroup{
				Name: crd.Spec.Group,
				Versions: []metav1.GroupVersionForDiscovery{
					{Version: version},
				},
				PreferredVersion: metav1.GroupVersionForDiscovery{Version: version},
			},
			VersionedResources: map[string][]metav1.APIResource{
				version: groupResource.APIResources,
			},
		},
	}

	restMapper := restmapper.NewDiscoveryRESTMapper(resources)
	resolver := scale.NewDiscoveryScaleKindResolver(discoveryClient)

	return scale.NewForConfig(config, restMapper, dynamic.LegacyAPIPathResolverFunc, resolver)
}
