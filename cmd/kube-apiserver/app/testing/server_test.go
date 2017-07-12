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

package testing

import (
	"fmt"
	"testing"
	"time"

	appsv1beta1 "k8s.io/api/apps/v1beta1"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
)

func TestRun(t *testing.T) {
	config, tearDown := StartTestServerOrDie(t)
	defer tearDown()

	client, err := kubernetes.NewForConfig(config)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// test whether the server is really healthy after /healthz told us so
	t.Logf("Creating Deployment directly after being healthy")
	var replicas int32 = 1
	_, err = client.AppsV1beta1().Deployments("default").Create(&appsv1beta1.Deployment{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Deployment",
			APIVersion: "apps/v1beta1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "test",
		},
		Spec: appsv1beta1.DeploymentSpec{
			Replicas: &replicas,
			Strategy: appsv1beta1.DeploymentStrategy{
				Type: appsv1beta1.RollingUpdateDeploymentStrategyType,
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"foo": "bar"},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "foo",
							Image: "foo",
						},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("Failed to create deployment: %v", err)
	}
}

func TestCRDShadowGroup(t *testing.T) {
	config, tearDown := StartTestServerOrDie(t)
	defer tearDown()

	kubeclient, err := kubernetes.NewForConfig(config)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	apiextensionsclient, err := apiextensionsclientset.NewForConfig(config)
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
	config, tearDown := StartTestServerOrDie(t)
	defer tearDown()

	apiextensionsclient, err := apiextensionsclientset.NewForConfig(config)
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
	barComConfig := *config
	barComConfig.GroupVersion = &schema.GroupVersion{Group: "cr.bar.com", Version: "v1"}
	barComConfig.APIPath = "/apis"
	barComClient, err := dynamic.NewClient(&barComConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	_, err = barComClient.Resource(&metav1.APIResource{Name: "foos", Namespaced: true}, "default").List(metav1.ListOptions{})
	if err != nil {
		t.Errorf("Failed to list foos.cr.bar.com instances: %v", err)
	}
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
