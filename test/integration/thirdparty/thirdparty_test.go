// +build integration,!no-etcd

/*
Copyright 2015 The Kubernetes Authors.

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

package thirdparty

// This file contains tests for the storage classes API resource.

import (
	"encoding/json"
	"reflect"
	"sort"
	"sync"
	"testing"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/wait"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestThirdPartyDiscovery(t *testing.T) {
	group := "company.com"
	version := "v1"

	_, s := framework.RunAMaster(framework.NewIntegrationTestMasterConfig())
	defer s.Close()
	clientConfig := &restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{NegotiatedSerializer: api.Codecs}}
	client := clientset.NewForConfigOrDie(clientConfig)

	// install thirdparty resource
	once := sync.Once{}
	deleteFoo := installThirdParty(t, client, clientConfig,
		&extensions.ThirdPartyResource{
			ObjectMeta: metav1.ObjectMeta{Name: "foo.company.com"},
			Versions:   []extensions.APIVersion{{Name: version}},
		}, group, version, "foos",
	)
	defer once.Do(deleteFoo)

	// check whether it shows up in discovery properly
	resources, err := client.Discovery().ServerResourcesForGroupVersion("company.com/" + version)
	if err != nil {
		t.Fatal(err)
	}
	if len(resources.APIResources) != 1 {
		t.Fatalf("Expected exactly the resource \"foos\" in group version %v/%v via discovery, got: %v", group, version, resources.APIResources)
	}
	r := resources.APIResources[0]
	if r.Name != "foos" {
		t.Fatalf("Expected exactly the resource \"foos\" in group version %v/%v via discovery, got: %v", group, version, r)
	}
	sort.Strings(r.Verbs)
	expectedVerbs := []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"}
	if !reflect.DeepEqual([]string(r.Verbs), expectedVerbs) {
		t.Fatalf("Unexpected verbs for resource \"foos\" in group version %v/%v via discovery: expected=%v got=%v", group, version, expectedVerbs, r.Verbs)
	}

	// delete
	once.Do(deleteFoo)

	// check whether resource is also gone from discovery
	resources, err = client.Discovery().ServerResourcesForGroupVersion(group + "/" + version)
	if err == nil {
		for _, r := range resources.APIResources {
			if r.Name == "foos" {
				t.Fatalf("unexpected resource \"foos\" in group version %v/%v after deletion", group, version)
			}
		}
	}
}

// TODO these tests will eventually be runnable in a single test
func TestThirdPartyDelete(t *testing.T) {
	_, s := framework.RunAMaster(framework.NewIntegrationTestMasterConfig())
	defer s.Close()

	clientConfig := &restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{NegotiatedSerializer: api.Codecs}}
	client := clientset.NewForConfigOrDie(clientConfig)

	DoTestInstallThirdPartyAPIDelete(t, client, clientConfig)
}

func TestThirdPartyMultiple(t *testing.T) {
	_, s := framework.RunAMaster(framework.NewIntegrationTestMasterConfig())
	defer s.Close()

	clientConfig := &restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{NegotiatedSerializer: api.Codecs}}
	client := clientset.NewForConfigOrDie(clientConfig)

	DoTestInstallMultipleAPIs(t, client, clientConfig)
}

// TODO make multiple versions work.  they've been broken
var versionsToTest = []string{"v1"}

type Foo struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty" description:"standard object metadata"`

	SomeField  string `json:"someField"`
	OtherField int    `json:"otherField"`
}

type FooList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" description:"standard list metadata; see http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata"`

	Items []Foo `json:"items"`
}

// installThirdParty installs a third party resource and returns a defer func
func installThirdParty(t *testing.T, client clientset.Interface, clientConfig *restclient.Config, tpr *extensions.ThirdPartyResource, group, version, resource string) func() {
	var err error
	_, err = client.Extensions().ThirdPartyResources().Create(tpr)
	if err != nil {
		t.Fatal(err)
	}

	fooClientConfig := *clientConfig
	fooClientConfig.APIPath = "apis"
	fooClientConfig.GroupVersion = &schema.GroupVersion{Group: group, Version: version}
	fooClient, err := restclient.RESTClientFor(&fooClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	err = wait.Poll(100*time.Millisecond, 60*time.Second, func() (bool, error) {
		_, err := fooClient.Get().Namespace("default").Resource(resource).DoRaw()
		if err == nil {
			return true, nil
		}
		if apierrors.IsNotFound(err) {
			return false, nil
		}

		return false, err
	})
	if err != nil {
		t.Fatal(err)
	}

	return func() {
		client.Extensions().ThirdPartyResources().Delete(tpr.Name, nil)
		err = wait.Poll(100*time.Millisecond, 60*time.Second, func() (bool, error) {
			_, err := fooClient.Get().Namespace("default").Resource(resource).DoRaw()
			if apierrors.IsNotFound(err) {
				return true, nil
			}

			return false, err
		})
		if err != nil {
			t.Fatal(err)
		}
	}
}

func DoTestInstallMultipleAPIs(t *testing.T, client clientset.Interface, clientConfig *restclient.Config) {
	group := "company.com"
	version := "v1"

	deleteFoo := installThirdParty(t, client, clientConfig,
		&extensions.ThirdPartyResource{
			ObjectMeta: metav1.ObjectMeta{Name: "foo.company.com"},
			Versions:   []extensions.APIVersion{{Name: version}},
		}, group, version, "foos",
	)
	defer deleteFoo()

	// TODO make multiple resources in one version work
	// deleteBar = installThirdParty(t, client, clientConfig,
	// 	&extensions.ThirdPartyResource{
	// 		ObjectMeta: metav1.ObjectMeta{Name: "bar.company.com"},
	// 		Versions:   []extensions.APIVersion{{Name: version}},
	// 	}, group, version, "bars",
	// )
	// defer deleteBar()
}

func DoTestInstallThirdPartyAPIDelete(t *testing.T, client clientset.Interface, clientConfig *restclient.Config) {
	for _, version := range versionsToTest {
		testInstallThirdPartyAPIDeleteVersion(t, client, clientConfig, version)
	}
}

func testInstallThirdPartyAPIDeleteVersion(t *testing.T, client clientset.Interface, clientConfig *restclient.Config, version string) {
	group := "company.com"

	deleteFoo := installThirdParty(t, client, clientConfig,
		&extensions.ThirdPartyResource{
			ObjectMeta: metav1.ObjectMeta{Name: "foo.company.com"},
			Versions:   []extensions.APIVersion{{Name: version}},
		}, group, version, "foos",
	)
	defer deleteFoo()

	fooClientConfig := *clientConfig
	fooClientConfig.APIPath = "apis"
	fooClientConfig.GroupVersion = &schema.GroupVersion{Group: group, Version: version}
	fooClient, err := restclient.RESTClientFor(&fooClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	expectedObj := Foo{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
		TypeMeta: metav1.TypeMeta{
			Kind: "Foo",
		},
		SomeField:  "test field",
		OtherField: 10,
	}
	objBytes, err := json.Marshal(&expectedObj)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := fooClient.Post().Namespace("default").Resource("foos").Body(objBytes).DoRaw(); err != nil {
		t.Fatal(err)
	}

	apiBytes, err := fooClient.Get().Namespace("default").Resource("foos").Name("test").DoRaw()
	if err != nil {
		t.Fatal(err)
	}
	item := Foo{}
	err = json.Unmarshal(apiBytes, &item)
	if err != nil {
		t.Fatal(err)
	}

	// Fill in fields set by the apiserver
	item.SelfLink = expectedObj.SelfLink
	item.ResourceVersion = expectedObj.ResourceVersion
	item.Namespace = expectedObj.Namespace
	item.UID = expectedObj.UID
	item.CreationTimestamp = expectedObj.CreationTimestamp
	if !reflect.DeepEqual(item, expectedObj) {
		t.Fatalf("expected:\n%v\n", diff.ObjectGoPrintSideBySide(expectedObj, item))
	}

	listBytes, err := fooClient.Get().Namespace("default").Resource("foos").DoRaw()
	if err != nil {
		t.Fatal(err)
	}
	list := FooList{}
	err = json.Unmarshal(listBytes, &list)
	if err != nil {
		t.Fatal(err)
	}
	if len(list.Items) != 1 {
		t.Fatalf("wrong item: %v", list)
	}

	if _, err := fooClient.Delete().Namespace("default").Resource("foos").Name("test").DoRaw(); err != nil {
		t.Fatal(err)
	}
	if _, err := fooClient.Get().Namespace("default").Resource("foos").Name("test").DoRaw(); !apierrors.IsNotFound(err) {
		t.Fatal(err)
	}
}
