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
	"fmt"
	"time"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
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

// NewRandomNameCustomResourceDefinition generates a CRD with random name to avoid name conflict in e2e tests
func NewRandomNameCustomResourceDefinition(scope apiextensionsv1beta1.ResourceScope) *apiextensionsv1beta1.CustomResourceDefinition {
	// ensure the singular doesn't end in an s for now
	gName := names.SimpleNameGenerator.GenerateName("foo") + "a"
	return &apiextensionsv1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: gName + "s.mygroup.example.com"},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group:   "mygroup.example.com",
			Version: "v1beta1",
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
				Plural:   gName + "s",
				Singular: gName,
				Kind:     gName,
				ListKind: gName + "List",
			},
			Scope: scope,
		},
	}
}

// NewNoxuCustomResourceDefinition returns a WishIHadChosenNoxu CRD.
func NewNoxuCustomResourceDefinition(scope apiextensionsv1beta1.ResourceScope) *apiextensionsv1beta1.CustomResourceDefinition {
	return &apiextensionsv1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: "noxus.mygroup.example.com"},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group:   "mygroup.example.com",
			Version: "v1beta1",
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
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
func NewMultipleVersionNoxuCRD(scope apiextensionsv1beta1.ResourceScope) *apiextensionsv1beta1.CustomResourceDefinition {
	return &apiextensionsv1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: "noxus.mygroup.example.com"},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group:   "mygroup.example.com",
			Version: "v1beta1",
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
				Plural:     "noxus",
				Singular:   "nonenglishnoxu",
				Kind:       "WishIHadChosenNoxu",
				ShortNames: []string{"foo", "bar", "abc", "def"},
				ListKind:   "NoxuItemList",
				Categories: []string{"all"},
			},
			Scope: scope,
			Versions: []apiextensionsv1beta1.CustomResourceDefinitionVersion{
				{
					Name:    "v1beta1",
					Served:  true,
					Storage: false,
				},
				{
					Name:    "v1beta2",
					Served:  true,
					Storage: true,
				},
				{
					Name:    "v0",
					Served:  false,
					Storage: false,
				},
			},
		},
	}
}

// NewNoxu2CustomResourceDefinition returns a WishIHadChosenNoxu2 CRD.
func NewNoxu2CustomResourceDefinition(scope apiextensionsv1beta1.ResourceScope) *apiextensionsv1beta1.CustomResourceDefinition {
	return &apiextensionsv1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: "noxus2.mygroup.example.com"},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group:   "mygroup.example.com",
			Version: "v1alpha1",
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
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

// NewCurletCustomResourceDefinition returns a Curlet CRD.
func NewCurletCustomResourceDefinition(scope apiextensionsv1beta1.ResourceScope) *apiextensionsv1beta1.CustomResourceDefinition {
	return &apiextensionsv1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: "curlets.mygroup.example.com"},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group:   "mygroup.example.com",
			Version: "v1beta1",
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
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

func servedVersions(crd *apiextensionsv1beta1.CustomResourceDefinition) []string {
	if len(crd.Spec.Versions) == 0 {
		return []string{crd.Spec.Version}
	}
	var versions []string
	for _, v := range crd.Spec.Versions {
		if v.Served {
			versions = append(versions, v.Name)
		}
	}
	return versions
}

func existsInDiscovery(crd *apiextensionsv1beta1.CustomResourceDefinition, apiExtensionsClient clientset.Interface, version string) (bool, error) {
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

// CreateNewCustomResourceDefinitionWatchUnsafe creates the CRD and makes sure
// the apiextension apiserver has installed the CRD. But it's not safe to watch
// the created CR. Please call CreateNewCustomResourceDefinition if you need to
// watch the CR.
func CreateNewCustomResourceDefinitionWatchUnsafe(crd *apiextensionsv1beta1.CustomResourceDefinition, apiExtensionsClient clientset.Interface) (*apiextensionsv1beta1.CustomResourceDefinition, error) {
	crd, err := apiExtensionsClient.ApiextensionsV1beta1().CustomResourceDefinitions().Create(crd)
	if err != nil {
		return nil, err
	}

	// wait until all resources appears in discovery
	for _, version := range servedVersions(crd) {
		err := wait.PollImmediate(500*time.Millisecond, 30*time.Second, func() (bool, error) {
			return existsInDiscovery(crd, apiExtensionsClient, version)
		})
		if err != nil {
			return nil, err
		}
	}

	return crd, err
}

// CreateNewCustomResourceDefinition creates the given CRD and makes sure its watch cache is primed on the server.
func CreateNewCustomResourceDefinition(crd *apiextensionsv1beta1.CustomResourceDefinition, apiExtensionsClient clientset.Interface, dynamicClientSet dynamic.Interface) (*apiextensionsv1beta1.CustomResourceDefinition, error) {
	crd, err := CreateNewCustomResourceDefinitionWatchUnsafe(crd, apiExtensionsClient)
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
	err = wait.PollImmediate(500*time.Millisecond, 30*time.Second, func() (bool, error) {
		return isWatchCachePrimed(crd, dynamicClientSet)
	})
	if err != nil {
		return nil, err
	}
	return crd, nil
}

func resourceClientForVersion(crd *apiextensionsv1beta1.CustomResourceDefinition, dynamicClientSet dynamic.Interface, namespace, version string) dynamic.ResourceInterface {
	gvr := schema.GroupVersionResource{Group: crd.Spec.Group, Version: version, Resource: crd.Spec.Names.Plural}
	if crd.Spec.Scope != apiextensionsv1beta1.ClusterScoped {
		return dynamicClientSet.Resource(gvr).Namespace(namespace)
	}
	return dynamicClientSet.Resource(gvr)
}

// isWatchCachePrimed returns true if the watch is primed for an specified version of CRD watch
func isWatchCachePrimed(crd *apiextensionsv1beta1.CustomResourceDefinition, dynamicClientSet dynamic.Interface) (bool, error) {
	ns := ""
	if crd.Spec.Scope != apiextensionsv1beta1.ClusterScoped {
		ns = "aval"
	}

	versions := servedVersions(crd)
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
	createdInstance, err := resourceClient.Create(instance, metav1.CreateOptions{})
	if err != nil {
		return false, err
	}
	err = resourceClient.Delete(createdInstance.GetName(), nil)
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

// DeleteCustomResourceDefinition deletes a CRD and waits until it disappears from discovery.
func DeleteCustomResourceDefinition(crd *apiextensionsv1beta1.CustomResourceDefinition, apiExtensionsClient clientset.Interface) error {
	if err := apiExtensionsClient.ApiextensionsV1beta1().CustomResourceDefinitions().Delete(crd.Name, nil); err != nil {
		return err
	}
	for _, version := range servedVersions(crd) {
		err := wait.PollImmediate(500*time.Millisecond, 30*time.Second, func() (bool, error) {
			exists, err := existsInDiscovery(crd, apiExtensionsClient, version)
			return !exists, err
		})
		if err != nil {
			return err
		}
	}
	return nil
}

// CreateNewVersionedScaleClient returns a scale client.
func CreateNewVersionedScaleClient(crd *apiextensionsv1beta1.CustomResourceDefinition, config *rest.Config, version string) (scale.ScalesGetter, error) {
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
