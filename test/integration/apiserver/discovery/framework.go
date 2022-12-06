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
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"time"

	apiextensions "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	aggregator "k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset"

	apidiscoveryv2beta1 "k8s.io/api/apidiscovery/v2beta1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiregistrationv1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
)

const acceptV1JSON = "application/json"
const acceptV2JSON = "application/json;g=apidiscovery.k8s.io;v=v2beta1;as=APIGroupDiscoveryList"

type testClient interface {
	kubernetes.Interface
	aggregator.Interface
	apiextensions.Interface
	dynamic.Interface
}

// declarative framework for discovery integration tests
// each test has metadata and a list of actions which each must pass for the
// test to pass
type testCase struct {
	Name    string
	Actions []testAction
}

// interface defining a function that does something with the integration test
// api server and returns an error. the test fails if the error is non nil
type testAction interface {
	Do(ctx context.Context, client testClient) error
}

type cleaningAction interface {
	testAction
	Cleanup(ctx context.Context, client testClient) error
}

// apply an apiservice to the cluster
type applyAPIService apiregistrationv1.APIServiceSpec

type applyCRD apiextensionsv1.CustomResourceDefinitionSpec

type deleteObject struct {
	metav1.GroupVersionResource
	Namespace string
	Name      string
}

// Wait for groupversions to appear in v1 discovery
type waitForGroupVersionsV1 []metav1.GroupVersion

// Wait for groupversions to disappear from v2 discovery
type waitForAbsentGroupVersionsV1 []metav1.GroupVersion

// Wait for groupversions to appear in v2 discovery
type waitForGroupVersionsV2 []metav1.GroupVersion

// Wait for groupversions to disappear from v2 discovery
type waitForAbsentGroupVersionsV2 []metav1.GroupVersion

// Assert something about the current state of v2 discovery
type inlineAction func(ctx context.Context, client testClient) error

func (a applyAPIService) Do(ctx context.Context, client testClient) error {
	// using dynamic client since the typed client does not support `Apply`
	// operation?
	obj := &apiregistrationv1.APIService{
		ObjectMeta: metav1.ObjectMeta{
			Name: a.Version + "." + a.Group,
		},
		Spec: apiregistrationv1.APIServiceSpec(a),
	}

	unstructuredContent, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		return err
	}

	unstructedObject := &unstructured.Unstructured{}
	unstructedObject.SetUnstructuredContent(unstructuredContent)
	unstructedObject.SetGroupVersionKind(apiregistrationv1.SchemeGroupVersion.WithKind("APIService"))

	_, err = client.
		Resource(apiregistrationv1.SchemeGroupVersion.WithResource("apiservices")).
		Apply(ctx, obj.Name, unstructedObject, metav1.ApplyOptions{
			FieldManager: "test-manager",
		})

	return err
}

func (a applyAPIService) Cleanup(ctx context.Context, client testClient) error {
	name := a.Version + "." + a.Group
	err := client.ApiregistrationV1().APIServices().Delete(ctx, name, metav1.DeleteOptions{})

	if !errors.IsNotFound(err) {
		return err
	}

	err = wait.PollWithContext(
		ctx,
		250*time.Millisecond,
		1*time.Second,
		func(ctx context.Context) (done bool, err error) {
			_, err = client.ApiregistrationV1().APIServices().Get(ctx, name, metav1.GetOptions{})
			if err == nil {
				return false, nil
			}

			if !errors.IsNotFound(err) {
				return false, err
			}
			return true, nil
		},
	)

	if err != nil {
		return fmt.Errorf("error waiting for APIService %v to clean up: %w", name, err)
	}

	return nil
}

func (a applyCRD) Do(ctx context.Context, client testClient) error {
	// using dynamic client since the typed client does not support `Apply`
	// operation?
	name := a.Names.Plural + "." + a.Group
	obj := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec(a),
	}

	unstructuredContent, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		return err
	}

	unstructedObject := &unstructured.Unstructured{}
	unstructedObject.SetUnstructuredContent(unstructuredContent)
	unstructedObject.SetGroupVersionKind(apiextensionsv1.SchemeGroupVersion.WithKind("CustomResourceDefinition"))

	_, err = client.
		Resource(apiextensionsv1.SchemeGroupVersion.WithResource("customresourcedefinitions")).
		Apply(ctx, obj.Name, unstructedObject, metav1.ApplyOptions{
			FieldManager: "test-manager",
		})

	return err
}

func (a applyCRD) Cleanup(ctx context.Context, client testClient) error {
	name := a.Names.Plural + "." + a.Group
	err := client.ApiextensionsV1().CustomResourceDefinitions().Delete(ctx, name, metav1.DeleteOptions{})

	if !errors.IsNotFound(err) {
		return err
	}

	err = wait.PollWithContext(
		ctx,
		250*time.Millisecond,
		1*time.Second,
		func(ctx context.Context) (done bool, err error) {
			_, err = client.ApiextensionsV1().CustomResourceDefinitions().Get(ctx, name, metav1.GetOptions{})
			if err == nil {
				return false, nil
			}

			if !errors.IsNotFound(err) {
				return false, err
			}
			return true, nil
		},
	)

	if err != nil {
		return fmt.Errorf("error waiting for CRD %v to clean up: %w", name, err)
	}

	return nil
}

func (d deleteObject) Do(ctx context.Context, client testClient) error {
	if d.Namespace == "" {
		return client.Resource(schema.GroupVersionResource(d.GroupVersionResource)).
			Delete(ctx, d.Name, metav1.DeleteOptions{})
	} else {
		return client.Resource(schema.GroupVersionResource(d.GroupVersionResource)).
			Namespace(d.Namespace).
			Delete(ctx, d.Name, metav1.DeleteOptions{})
	}
}

func (w waitForGroupVersionsV2) Do(ctx context.Context, client testClient) error {
	err := WaitForResultWithCondition(ctx, client, func(result apidiscoveryv2beta1.APIGroupDiscoveryList) bool {
		for _, gv := range w {
			if FindGroupVersionV2(result, gv) == nil {
				return false
			}
		}

		return true
	})

	if err != nil {
		return fmt.Errorf("waiting for groupversions v2 (%v): %w", w, err)
	}
	return nil
}

func (w waitForAbsentGroupVersionsV2) Do(ctx context.Context, client testClient) error {
	err := WaitForResultWithCondition(ctx, client, func(result apidiscoveryv2beta1.APIGroupDiscoveryList) bool {
		for _, gv := range w {
			if FindGroupVersionV2(result, gv) != nil {
				return false
			}
		}

		return true
	})

	if err != nil {
		return fmt.Errorf("waiting for absent groupversions v2 (%v): %w", w, err)
	}
	return nil
}

func (w waitForGroupVersionsV1) Do(ctx context.Context, client testClient) error {
	err := WaitForV1GroupsWithCondition(ctx, client, func(result metav1.APIGroupList) bool {
		for _, gv := range w {
			if !FindGroupVersionV1(result, gv) {
				return false
			}
		}

		return true
	})

	if err != nil {
		return fmt.Errorf("waiting for groupversions v1 (%v): %w", w, err)
	}
	return nil
}

func (w waitForAbsentGroupVersionsV1) Do(ctx context.Context, client testClient) error {
	err := WaitForV1GroupsWithCondition(ctx, client, func(result metav1.APIGroupList) bool {
		for _, gv := range w {
			if FindGroupVersionV1(result, gv) {
				return false
			}
		}

		return true
	})

	if err != nil {
		return fmt.Errorf("waiting for absent groupversions v1 (%v): %w", w, err)
	}
	return nil
}

func (i inlineAction) Do(ctx context.Context, client testClient) error {
	return i(ctx, client)
}

func FetchV2Discovery(ctx context.Context, client testClient) (apidiscoveryv2beta1.APIGroupDiscoveryList, error) {
	result, err := client.
		Discovery().
		RESTClient().
		Get().
		AbsPath("/apis").
		SetHeader("Accept", acceptV2JSON).
		Do(ctx).
		Raw()

	if err != nil {
		return apidiscoveryv2beta1.APIGroupDiscoveryList{}, fmt.Errorf("failed to fetch v2 discovery: %w", err)
	}

	groupList := apidiscoveryv2beta1.APIGroupDiscoveryList{}
	err = json.Unmarshal(result, &groupList)
	if err != nil {
		return apidiscoveryv2beta1.APIGroupDiscoveryList{}, fmt.Errorf("failed to parse v2 discovery: %w", err)
	}

	return groupList, nil
}

func FetchV1DiscoveryGroups(ctx context.Context, client testClient) (metav1.APIGroupList, error) {
	return FetchV1DiscoveryGroupsAtPath(ctx, client, "/apis")
}

func FetchV1DiscoveryLegacyGroups(ctx context.Context, client testClient) (metav1.APIGroupList, error) {
	return FetchV1DiscoveryGroupsAtPath(ctx, client, "/api")
}

func FetchV1DiscoveryGroupsAtPath(ctx context.Context, client testClient, path string) (metav1.APIGroupList, error) {
	result, err := client.
		Discovery().
		RESTClient().
		Get().
		AbsPath("/apis").
		SetHeader("Accept", acceptV1JSON).
		Do(ctx).
		Raw()

	if err != nil {
		return metav1.APIGroupList{}, fmt.Errorf("failed to fetch v1 discovery at %v: %w", path, err)
	}

	groupList := metav1.APIGroupList{}
	err = json.Unmarshal(result, &groupList)
	if err != nil {
		return metav1.APIGroupList{}, fmt.Errorf("failed to parse v1 discovery at %v: %w", path, err)
	}

	return groupList, nil
}

func FetchV1DiscoveryResource(ctx context.Context, client testClient, gv metav1.GroupVersion) (metav1.APIGroupList, error) {
	result, err := client.
		Discovery().
		RESTClient().
		Get().
		AbsPath("/apis/"+gv.Group+"/"+gv.Version).
		SetHeader("Accept", acceptV1JSON).
		Do(ctx).
		Raw()

	if err != nil {
		return metav1.APIGroupList{}, err
	}

	groupList := metav1.APIGroupList{}
	err = json.Unmarshal(result, &groupList)
	if err != nil {
		return metav1.APIGroupList{}, err
	}

	return groupList, nil
}

func WaitForGroupsAbsent(ctx context.Context, client testClient, groups ...string) error {
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

func WaitForGroups(ctx context.Context, client testClient, groups ...apidiscoveryv2beta1.APIGroupDiscovery) error {
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

func WaitForResultWithCondition(ctx context.Context, client testClient, condition func(result apidiscoveryv2beta1.APIGroupDiscoveryList) bool) error {
	// Keep repeatedly fetching document from aggregator.
	// Check to see if it contains our service within a reasonable amount of time
	return wait.PollWithContext(
		ctx,
		250*time.Millisecond,
		1*time.Second,
		func(ctx context.Context) (done bool, err error) {
			groupList, err := FetchV2Discovery(ctx, client)
			if err != nil {
				return false, err
			}

			if condition(groupList) {
				return true, nil
			}

			return false, nil
		})
}

func WaitForV1GroupsWithCondition(ctx context.Context, client testClient, condition func(result metav1.APIGroupList) bool) error {
	// Keep repeatedly fetching document from aggregator.
	// Check to see if it contains our service within a reasonable amount of time
	return wait.PollWithContext(
		ctx,
		250*time.Millisecond,
		1*time.Second,
		func(ctx context.Context) (done bool, err error) {
			groupList, err := FetchV1DiscoveryGroups(ctx, client)

			if err != nil {
				return false, err
			}

			if condition(groupList) {
				return true, nil
			}

			return false, nil
		})
}

func FindGroupVersionV1(discovery metav1.APIGroupList, gv metav1.GroupVersion) bool {
	for _, documentGroup := range discovery.Groups {
		if documentGroup.Name != gv.Group {
			continue
		}

		for _, documentVersion := range documentGroup.Versions {
			if documentVersion.Version == gv.Version {
				return true
			}
		}
	}

	return false
}

func FindGroupVersionV2(discovery apidiscoveryv2beta1.APIGroupDiscoveryList, gv metav1.GroupVersion) *apidiscoveryv2beta1.APIVersionDiscovery {
	for _, documentGroup := range discovery.Items {
		if documentGroup.Name != gv.Group {
			continue
		}

		for _, documentVersion := range documentGroup.Versions {
			if documentVersion.Version == gv.Version {
				return &documentVersion
			}
		}
	}

	return nil
}
