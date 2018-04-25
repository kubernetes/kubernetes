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

package master

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"

	admissionregistrationv1alpha1 "k8s.io/api/admissionregistration/v1alpha1"
	networkingv1 "k8s.io/api/networking/v1"
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestCRDShadowGroup(t *testing.T) {
	result := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.SharedEtcd())
	defer result.TearDownFn()

	kubeclient, err := kubernetes.NewForConfig(result.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	apiextensionsclient, err := apiextensionsclientset.NewForConfig(result.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	t.Logf("Creating a NetworkPolicy")
	nwPolicy, err := kubeclient.NetworkingV1().NetworkPolicies("default").Create(&networkingv1.NetworkPolicy{
		ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
		Spec: networkingv1.NetworkPolicySpec{
			PodSelector: metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
			Ingress:     []networkingv1.NetworkPolicyIngressRule{},
		},
	})
	if err != nil {
		t.Fatalf("Failed to create NetworkPolicy: %v", err)
	}

	t.Logf("Trying to shadow networking group")
	crd := &apiextensionsv1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foos." + networkingv1.GroupName,
		},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group:   networkingv1.GroupName,
			Version: networkingv1.SchemeGroupVersion.Version,
			Scope:   apiextensionsv1beta1.ClusterScoped,
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
				Plural: "foos",
				Kind:   "Foo",
			},
		},
	}
	if _, err = apiextensionsclient.ApiextensionsV1beta1().CustomResourceDefinitions().Create(crd); err != nil {
		t.Fatalf("Failed to create networking group CRD: %v", err)
	}
	if err := waitForEstablishedCRD(apiextensionsclient, crd.Name); err != nil {
		t.Fatalf("Failed to establish networking group CRD: %v", err)
	}
	// wait to give aggregator time to update
	time.Sleep(2 * time.Second)

	t.Logf("Checking that we still see the NetworkPolicy")
	_, err = kubeclient.NetworkingV1().NetworkPolicies(nwPolicy.Namespace).Get(nwPolicy.Name, metav1.GetOptions{})
	if err != nil {
		t.Errorf("Failed to get NetworkPolocy: %v", err)
	}

	t.Logf("Checking that crd resource does not show up in networking group")
	found, err := crdExistsInDiscovery(apiextensionsclient, crd)
	if err != nil {
		t.Fatalf("unexpected discovery error: %v", err)
	}
	if found {
		t.Errorf("CRD resource shows up in discovery, but shouldn't.")
	}
}

func TestCRD(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.Initializers, true)()

	result := kubeapiservertesting.StartTestServerOrDie(t, []string{"--admission-control", "Initializers"}, framework.SharedEtcd())
	defer result.TearDownFn()

	kubeclient, err := kubernetes.NewForConfig(result.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	apiextensionsclient, err := apiextensionsclientset.NewForConfig(result.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	t.Logf("Trying to create a custom resource without conflict")
	crd := &apiextensionsv1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foos.cr.bar.com",
		},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group:   "cr.bar.com",
			Version: "v1",
			Scope:   apiextensionsv1beta1.NamespaceScoped,
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
				Plural: "foos",
				Kind:   "Foo",
			},
		},
	}
	if _, err = apiextensionsclient.ApiextensionsV1beta1().CustomResourceDefinitions().Create(crd); err != nil {
		t.Fatalf("Failed to create foos.cr.bar.com CRD; %v", err)
	}
	if err := waitForEstablishedCRD(apiextensionsclient, crd.Name); err != nil {
		t.Fatalf("Failed to establish foos.cr.bar.com CRD: %v", err)
	}
	if err := wait.PollImmediate(500*time.Millisecond, 30*time.Second, func() (bool, error) {
		return crdExistsInDiscovery(apiextensionsclient, crd)
	}); err != nil {
		t.Fatalf("Failed to see foos.cr.bar.com in discovery: %v", err)
	}

	t.Logf("Trying to access foos.cr.bar.com with dynamic client")
	barComConfig := *result.ClientConfig
	barComConfig.GroupVersion = &schema.GroupVersion{Group: "cr.bar.com", Version: "v1"}
	barComConfig.APIPath = "/apis"
	barComClient, err := dynamic.NewClient(&barComConfig, *barComConfig.GroupVersion)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	_, err = barComClient.Resource(&metav1.APIResource{Name: "foos", Namespaced: true}, "default").List(metav1.ListOptions{})
	if err != nil {
		t.Errorf("Failed to list foos.cr.bar.com instances: %v", err)
	}

	t.Logf("Creating InitializerConfiguration")
	_, err = kubeclient.AdmissionregistrationV1alpha1().InitializerConfigurations().Create(&admissionregistrationv1alpha1.InitializerConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foos.cr.bar.com",
		},
		Initializers: []admissionregistrationv1alpha1.Initializer{
			{
				Name: "cr.bar.com",
				Rules: []admissionregistrationv1alpha1.Rule{
					{
						APIGroups:   []string{"cr.bar.com"},
						APIVersions: []string{"*"},
						Resources:   []string{"*"},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("Failed to create InitializerConfiguration: %v", err)
	}

	// TODO DO NOT MERGE THIS
	time.Sleep(5 * time.Second)

	t.Logf("Creating Foo instance")
	foo := &Foo{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "cr.bar.com/v1",
			Kind:       "Foo",
		},
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
	}
	unstructuredFoo, err := unstructuredFoo(foo)
	if err != nil {
		t.Fatalf("Unable to create Foo: %v", err)
	}
	createErr := make(chan error, 1)
	go func() {
		_, err := barComClient.Resource(&metav1.APIResource{Name: "foos", Namespaced: true}, "default").Create(unstructuredFoo)
		t.Logf("Foo instance create returned: %v", err)
		if err != nil {
			createErr <- err
		}
	}()

	err = wait.PollImmediate(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		select {
		case createErr := <-createErr:
			return true, createErr
		default:
		}

		t.Logf("Checking that Foo instance is visible with IncludeUninitialized=true")
		_, err := barComClient.Resource(&metav1.APIResource{Name: "foos", Namespaced: true}, "default").Get(foo.ObjectMeta.Name, metav1.GetOptions{
			IncludeUninitialized: true,
		})
		switch {
		case err == nil:
			return true, nil
		case errors.IsNotFound(err):
			return false, nil
		default:
			return false, err
		}
	})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	t.Logf("Removing initializer from Foo instance")
	success := false
	for i := 0; i < 10; i++ {
		// would love to replace the following with a patch, but removing strings from the intitializer array
		// is not what JSON (Merge) patch authors had in mind.
		fooUnstructured, err := barComClient.Resource(&metav1.APIResource{Name: "foos", Namespaced: true}, "default").Get(foo.ObjectMeta.Name, metav1.GetOptions{
			IncludeUninitialized: true,
		})
		if err != nil {
			t.Fatalf("Error getting Foo instance: %v", err)
		}
		bs, _ := fooUnstructured.MarshalJSON()
		t.Logf("Got Foo instance: %v", string(bs))
		foo := Foo{}
		if err := json.Unmarshal(bs, &foo); err != nil {
			t.Fatalf("Error parsing Foo instance: %v", err)
		}

		// remove initialize
		if foo.ObjectMeta.Initializers == nil {
			t.Fatalf("Expected initializers to be set in Foo instance")
		}
		found := false
		for i := range foo.ObjectMeta.Initializers.Pending {
			if foo.ObjectMeta.Initializers.Pending[i].Name == "cr.bar.com" {
				foo.ObjectMeta.Initializers.Pending = append(foo.ObjectMeta.Initializers.Pending[:i], foo.ObjectMeta.Initializers.Pending[i+1:]...)
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("Expected cr.bar.com as initializer on Foo instance")
		}
		if len(foo.ObjectMeta.Initializers.Pending) == 0 && foo.ObjectMeta.Initializers.Result == nil {
			foo.ObjectMeta.Initializers = nil
		}
		bs, err = json.Marshal(&foo)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		fooUnstructured.UnmarshalJSON(bs)

		_, err = barComClient.Resource(&metav1.APIResource{Name: "foos", Namespaced: true}, "default").Update(fooUnstructured)
		if err != nil && !errors.IsConflict(err) {
			t.Fatalf("Failed to update Foo instance: %v", err)
		} else if err == nil {
			success = true
			break
		}
	}
	if !success {
		t.Fatalf("Failed to remove initializer from Foo object")
	}

	t.Logf("Checking that Foo instance is visible after removing the initializer")
	if _, err := barComClient.Resource(&metav1.APIResource{Name: "foos", Namespaced: true}, "default").Get(foo.ObjectMeta.Name, metav1.GetOptions{}); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

type Foo struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
}

func unstructuredFoo(foo *Foo) (*unstructured.Unstructured, error) {
	bs, err := json.Marshal(foo)
	if err != nil {
		return nil, err
	}
	ret := &unstructured.Unstructured{}
	if err = ret.UnmarshalJSON(bs); err != nil {
		return nil, err
	}
	return ret, nil
}

func waitForEstablishedCRD(client apiextensionsclientset.Interface, name string) error {
	return wait.PollImmediate(500*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		crd, err := client.ApiextensionsV1beta1().CustomResourceDefinitions().Get(name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		for _, cond := range crd.Status.Conditions {
			switch cond.Type {
			case apiextensionsv1beta1.Established:
				if cond.Status == apiextensionsv1beta1.ConditionTrue {
					return true, err
				}
			case apiextensionsv1beta1.NamesAccepted:
				if cond.Status == apiextensionsv1beta1.ConditionFalse {
					fmt.Printf("Name conflict: %v\n", cond.Reason)
				}
			}
		}
		return false, nil
	})
}

func crdExistsInDiscovery(client apiextensionsclientset.Interface, crd *apiextensionsv1beta1.CustomResourceDefinition) (bool, error) {
	resourceList, err := client.Discovery().ServerResourcesForGroupVersion(crd.Spec.Group + "/" + crd.Spec.Version)
	if err != nil {
		return false, nil
	}
	for _, resource := range resourceList.APIResources {
		if resource.Name == crd.Spec.Names.Plural {
			return true, nil
		}
	}
	return false, nil
}
