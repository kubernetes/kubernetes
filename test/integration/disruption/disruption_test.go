/*
Copyright 2019 The Kubernetes Authors.

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

package disruption

import (
	"fmt"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/api/policy/v1beta1"
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	cacheddiscovery "k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/scale"
	"k8s.io/client-go/tools/cache"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/disruption"
	"k8s.io/kubernetes/test/integration/etcd"
	"k8s.io/kubernetes/test/integration/framework"
)

func setup(t *testing.T) (*kubeapiservertesting.TestServer, *disruption.DisruptionController, informers.SharedInformerFactory, clientset.Interface, *apiextensionsclientset.Clientset, dynamic.Interface) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--disable-admission-plugins", "ServiceAccount"}, framework.SharedEtcd())

	clientSet, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}
	resyncPeriod := 12 * time.Hour
	informers := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(server.ClientConfig, "pdb-informers")), resyncPeriod)

	client := clientset.NewForConfigOrDie(restclient.AddUserAgent(server.ClientConfig, "disruption-controller"))

	discoveryClient := cacheddiscovery.NewMemCacheClient(clientSet.Discovery())
	mapper := restmapper.NewDeferredDiscoveryRESTMapper(discoveryClient)

	scaleKindResolver := scale.NewDiscoveryScaleKindResolver(client.Discovery())
	scaleClient, err := scale.NewForConfig(server.ClientConfig, mapper, dynamic.LegacyAPIPathResolverFunc, scaleKindResolver)
	if err != nil {
		t.Fatalf("Error creating scaleClient: %v", err)
	}

	apiExtensionClient, err := apiextensionsclientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating extension clientset: %v", err)
	}

	dynamicClient, err := dynamic.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("Error creating dynamicClient: %v", err)
	}

	pdbc := disruption.NewDisruptionController(
		informers.Core().V1().Pods(),
		informers.Policy().V1beta1().PodDisruptionBudgets(),
		informers.Core().V1().ReplicationControllers(),
		informers.Apps().V1().ReplicaSets(),
		informers.Apps().V1().Deployments(),
		informers.Apps().V1().StatefulSets(),
		client,
		mapper,
		scaleClient,
	)
	return server, pdbc, informers, clientSet, apiExtensionClient, dynamicClient
}

func TestPDBWithScaleSubresource(t *testing.T) {
	s, pdbc, informers, clientSet, apiExtensionClient, dynamicClient := setup(t)
	defer s.TearDownFn()

	nsName := "pdb-scale-subresource"
	createNs(t, nsName, clientSet)

	stopCh := make(chan struct{})
	informers.Start(stopCh)
	go pdbc.Run(stopCh)
	defer close(stopCh)

	crdDefinition := newCustomResourceDefinition()
	etcd.CreateTestCRDs(t, apiExtensionClient, true, crdDefinition)
	gvr := schema.GroupVersionResource{Group: crdDefinition.Spec.Group, Version: crdDefinition.Spec.Version, Resource: crdDefinition.Spec.Names.Plural}
	resourceClient := dynamicClient.Resource(gvr).Namespace(nsName)

	replicas := 4
	maxUnavailable := int32(2)
	podLabelValue := "test-crd"

	resource := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       crdDefinition.Spec.Names.Kind,
			"apiVersion": crdDefinition.Spec.Group + "/" + crdDefinition.Spec.Version,
			"metadata": map[string]interface{}{
				"name":      "resource",
				"namespace": nsName,
			},
			"spec": map[string]interface{}{
				"replicas": replicas,
			},
		},
	}
	createdResource, err := resourceClient.Create(resource, metav1.CreateOptions{})
	if err != nil {
		t.Error(err)
	}

	trueValue := true
	ownerRef := metav1.OwnerReference{
		Name:       resource.GetName(),
		Kind:       crdDefinition.Spec.Names.Kind,
		APIVersion: crdDefinition.Spec.Group + "/" + crdDefinition.Spec.Version,
		UID:        createdResource.GetUID(),
		Controller: &trueValue,
	}
	for i := 0; i < replicas; i++ {
		createPod(t, fmt.Sprintf("pod-%d", i), nsName, podLabelValue, clientSet, ownerRef)
	}

	waitToObservePods(t, informers.Core().V1().Pods().Informer(), 4, v1.PodRunning)

	pdb := &v1beta1.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-pdb",
		},
		Spec: v1beta1.PodDisruptionBudgetSpec{
			MaxUnavailable: &intstr.IntOrString{
				Type:   intstr.Int,
				IntVal: maxUnavailable,
			},
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": podLabelValue},
			},
		},
	}
	if _, err := clientSet.PolicyV1beta1().PodDisruptionBudgets(nsName).Create(pdb); err != nil {
		t.Errorf("Error creating PodDisruptionBudget: %v", err)
	}

	waitPDBStable(t, clientSet, 4, nsName, pdb.Name)

	newPdb, err := clientSet.PolicyV1beta1().PodDisruptionBudgets(nsName).Get(pdb.Name, metav1.GetOptions{})

	if expected, found := int32(replicas), newPdb.Status.ExpectedPods; expected != found {
		t.Errorf("Expected %d, but found %d", expected, found)
	}
	if expected, found := int32(replicas)-maxUnavailable, newPdb.Status.DesiredHealthy; expected != found {
		t.Errorf("Expected %d, but found %d", expected, found)
	}
	if expected, found := maxUnavailable, newPdb.Status.PodDisruptionsAllowed; expected != found {
		t.Errorf("Expected %d, but found %d", expected, found)
	}
}

func createPod(t *testing.T, name, namespace, labelValue string, clientSet clientset.Interface, ownerRef metav1.OwnerReference) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels:    map[string]string{"app": labelValue},
			OwnerReferences: []metav1.OwnerReference{
				ownerRef,
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "fake-name",
					Image: "fakeimage",
				},
			},
		},
	}
	_, err := clientSet.CoreV1().Pods(namespace).Create(pod)
	if err != nil {
		t.Error(err)
	}
	addPodConditionReady(pod)
	if _, err := clientSet.CoreV1().Pods(namespace).UpdateStatus(pod); err != nil {
		t.Error(err)
	}
}

func createNs(t *testing.T, name string, clientSet clientset.Interface) {
	_, err := clientSet.CoreV1().Namespaces().Create(&v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
	})
	if err != nil {
		t.Errorf("Error creating namespace: %v", err)
	}
}

func addPodConditionReady(pod *v1.Pod) {
	pod.Status = v1.PodStatus{
		Phase: v1.PodRunning,
		Conditions: []v1.PodCondition{
			{
				Type:   v1.PodReady,
				Status: v1.ConditionTrue,
			},
		},
	}
}

func newCustomResourceDefinition() *apiextensionsv1beta1.CustomResourceDefinition {
	return &apiextensionsv1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: "crds.mygroup.example.com"},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group:   "mygroup.example.com",
			Version: "v1beta1",
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
				Plural:   "crds",
				Singular: "crd",
				Kind:     "Crd",
				ListKind: "CrdList",
			},
			Scope: apiextensionsv1beta1.NamespaceScoped,
			Subresources: &apiextensionsv1beta1.CustomResourceSubresources{
				Scale: &apiextensionsv1beta1.CustomResourceSubresourceScale{
					SpecReplicasPath:   ".spec.replicas",
					StatusReplicasPath: ".status.replicas",
				},
			},
		},
	}
}

func waitPDBStable(t *testing.T, clientSet clientset.Interface, podNum int32, ns, pdbName string) {
	if err := wait.PollImmediate(2*time.Second, 60*time.Second, func() (bool, error) {
		pdb, err := clientSet.PolicyV1beta1().PodDisruptionBudgets(ns).Get(pdbName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if pdb.Status.CurrentHealthy != podNum {
			return false, nil
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func waitToObservePods(t *testing.T, podInformer cache.SharedIndexInformer, podNum int, phase v1.PodPhase) {
	if err := wait.PollImmediate(2*time.Second, 60*time.Second, func() (bool, error) {
		objects := podInformer.GetIndexer().List()
		if len(objects) != podNum {
			return false, nil
		}
		for _, obj := range objects {
			pod := obj.(*v1.Pod)
			if pod.Status.Phase != phase {
				return false, nil
			}
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}
