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

package garbagecollector

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	apiextensionstestserver "k8s.io/apiextensions-apiserver/test/integration/fixtures"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/storage/names"
	cacheddiscovery "k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/metadata"
	"k8s.io/client-go/metadata/metadatainformer"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/tools/cache"
	"k8s.io/controller-manager/pkg/informerfactory"
	"k8s.io/klog/v2"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/garbagecollector"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

func getForegroundOptions() metav1.DeleteOptions {
	policy := metav1.DeletePropagationForeground
	return metav1.DeleteOptions{PropagationPolicy: &policy}
}

func getOrphanOptions() metav1.DeleteOptions {
	var trueVar = true
	return metav1.DeleteOptions{OrphanDependents: &trueVar}
}

func getPropagateOrphanOptions() metav1.DeleteOptions {
	policy := metav1.DeletePropagationOrphan
	return metav1.DeleteOptions{PropagationPolicy: &policy}
}

func getNonOrphanOptions() metav1.DeleteOptions {
	var falseVar = false
	return metav1.DeleteOptions{OrphanDependents: &falseVar}
}

const garbageCollectedPodName = "test.pod.1"
const independentPodName = "test.pod.2"
const oneValidOwnerPodName = "test.pod.3"
const toBeDeletedRCName = "test.rc.1"
const remainingRCName = "test.rc.2"

// testCert was generated from crypto/tls/generate_cert.go with the following command:
//
//	go run generate_cert.go  --rsa-bits 2048 --host 127.0.0.1,::1,example.com --ca --start-date "Jan 1 00:00:00 1970" --duration=1000000h
var testCert = []byte(`-----BEGIN CERTIFICATE-----
MIIDGDCCAgCgAwIBAgIQTKCKn99d5HhQVCLln2Q+eTANBgkqhkiG9w0BAQsFADAS
MRAwDgYDVQQKEwdBY21lIENvMCAXDTcwMDEwMTAwMDAwMFoYDzIwODQwMTI5MTYw
MDAwWjASMRAwDgYDVQQKEwdBY21lIENvMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8A
MIIBCgKCAQEA1Z5/aTwqY706M34tn60l8ZHkanWDl8mM1pYf4Q7qg3zA9XqWLX6S
4rTYDYCb4stEasC72lQnbEWHbthiQE76zubP8WOFHdvGR3mjAvHWz4FxvLOTheZ+
3iDUrl6Aj9UIsYqzmpBJAoY4+vGGf+xHvuukHrVcFqR9ZuBdZuJ/HbbjUyuNr3X9
erNIr5Ha17gVzf17SNbYgNrX9gbCeEB8Z9Ox7dVuJhLDkpF0T/B5Zld3BjyUVY/T
cukU4dTVp6isbWPvCMRCZCCOpb+qIhxEjJ0n6tnPt8nf9lvDl4SWMl6X1bH+2EFa
a8R06G0QI+XhwPyjXUyCR8QEOZPCR5wyqQIDAQABo2gwZjAOBgNVHQ8BAf8EBAMC
AqQwEwYDVR0lBAwwCgYIKwYBBQUHAwEwDwYDVR0TAQH/BAUwAwEB/zAuBgNVHREE
JzAlggtleGFtcGxlLmNvbYcEfwAAAYcQAAAAAAAAAAAAAAAAAAAAATANBgkqhkiG
9w0BAQsFAAOCAQEAThqgJ/AFqaANsOp48lojDZfZBFxJQ3A4zfR/MgggUoQ9cP3V
rxuKAFWQjze1EZc7J9iO1WvH98lOGVNRY/t2VIrVoSsBiALP86Eew9WucP60tbv2
8/zsBDSfEo9Wl+Q/gwdEh8dgciUKROvCm76EgAwPGicMAgRsxXgwXHhS5e8nnbIE
Ewaqvb5dY++6kh0Oz+adtNT5OqOwXTIRI67WuEe6/B3Z4LNVPQDIj7ZUJGNw8e6L
F4nkUthwlKx4yEJHZBRuFPnO7Z81jNKuwL276+mczRH7piI6z9uyMV/JbEsOIxyL
W6CzB7pZ9Nj1YLpgzc1r6oONHLokMJJIz/IvkQ==
-----END CERTIFICATE-----`)

func newPod(podName, podNamespace string, ownerReferences []metav1.OwnerReference) *v1.Pod {
	for i := 0; i < len(ownerReferences); i++ {
		if len(ownerReferences[i].Kind) == 0 {
			ownerReferences[i].Kind = "ReplicationController"
		}
		ownerReferences[i].APIVersion = "v1"
	}
	return &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:            podName,
			Namespace:       podNamespace,
			OwnerReferences: ownerReferences,
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
}

func newOwnerRC(name, namespace string) *v1.ReplicationController {
	return &v1.ReplicationController{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ReplicationController",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Spec: v1.ReplicationControllerSpec{
			Selector: map[string]string{"name": "test"},
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"name": "test"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "fake-name",
							Image: "fakeimage",
						},
					},
				},
			},
		},
	}
}

func newCRDInstance(definition *apiextensionsv1.CustomResourceDefinition, namespace, name string) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       definition.Spec.Names.Kind,
			"apiVersion": definition.Spec.Group + "/" + definition.Spec.Versions[0].Name,
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
		},
	}
}

func newConfigMap(namespace, name string) *v1.ConfigMap {
	return &v1.ConfigMap{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ConfigMap",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
	}
}

func link(t *testing.T, owner, dependent metav1.Object) {
	ownerType, err := meta.TypeAccessor(owner)
	if err != nil {
		t.Fatalf("failed to get type info for %#v: %v", owner, err)
	}
	ref := metav1.OwnerReference{
		Kind:       ownerType.GetKind(),
		APIVersion: ownerType.GetAPIVersion(),
		Name:       owner.GetName(),
		UID:        owner.GetUID(),
	}
	dependent.SetOwnerReferences(append(dependent.GetOwnerReferences(), ref))
}

func createRandomCustomResourceDefinition(
	t *testing.T, apiExtensionClient apiextensionsclientset.Interface,
	dynamicClient dynamic.Interface,
	namespace string,
) (*apiextensionsv1.CustomResourceDefinition, dynamic.ResourceInterface) {
	// Create a random custom resource definition and ensure it's available for
	// use.
	definition := apiextensionstestserver.NewRandomNameV1CustomResourceDefinition(apiextensionsv1.NamespaceScoped)

	definition, err := apiextensionstestserver.CreateNewV1CustomResourceDefinition(definition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatalf("failed to create CustomResourceDefinition: %v", err)
	}

	// Get a client for the custom resource.
	gvr := schema.GroupVersionResource{Group: definition.Spec.Group, Version: definition.Spec.Versions[0].Name, Resource: definition.Spec.Names.Plural}

	resourceClient := dynamicClient.Resource(gvr).Namespace(namespace)

	return definition, resourceClient
}

type testContext struct {
	logger             klog.Logger
	tearDown           func()
	gc                 *garbagecollector.GarbageCollector
	clientSet          clientset.Interface
	apiExtensionClient apiextensionsclientset.Interface
	dynamicClient      dynamic.Interface
	metadataClient     metadata.Interface
	startGC            func(workers int)
	// syncPeriod is how often the GC started with startGC will be resynced.
	syncPeriod time.Duration
}

// if workerCount > 0, will start the GC, otherwise it's up to the caller to Run() the GC.
func setup(t *testing.T, workerCount int) *testContext {
	return setupWithServer(t, kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd()), workerCount)
}

func setupWithServer(t *testing.T, result *kubeapiservertesting.TestServer, workerCount int) *testContext {
	clientSet, err := clientset.NewForConfig(result.ClientConfig)
	if err != nil {
		t.Fatalf("error creating clientset: %v", err)
	}

	// Helpful stuff for testing CRD.
	apiExtensionClient, err := apiextensionsclientset.NewForConfig(result.ClientConfig)
	if err != nil {
		t.Fatalf("error creating extension clientset: %v", err)
	}
	// CreateCRDUsingRemovedAPI wants to use this namespace for verifying
	// namespace-scoped CRD creation.
	createNamespaceOrDie("aval", clientSet, t)

	discoveryClient := cacheddiscovery.NewMemCacheClient(clientSet.Discovery())
	restMapper := restmapper.NewDeferredDiscoveryRESTMapper(discoveryClient)
	restMapper.Reset()
	config := *result.ClientConfig
	metadataClient, err := metadata.NewForConfig(&config)
	if err != nil {
		t.Fatalf("failed to create metadataClient: %v", err)
	}
	dynamicClient, err := dynamic.NewForConfig(&config)
	if err != nil {
		t.Fatalf("failed to create dynamicClient: %v", err)
	}
	sharedInformers := informers.NewSharedInformerFactory(clientSet, 0)
	metadataInformers := metadatainformer.NewSharedInformerFactory(metadataClient, 0)

	tCtx := ktesting.Init(t)
	logger := tCtx.Logger()
	alwaysStarted := make(chan struct{})
	close(alwaysStarted)

	gc, err := garbagecollector.NewGarbageCollector(
		tCtx,
		clientSet,
		metadataClient,
		restMapper,
		garbagecollector.DefaultIgnoredResources(),
		informerfactory.NewInformerFactory(sharedInformers, metadataInformers),
		alwaysStarted,
	)
	if err != nil {
		t.Fatalf("failed to create garbage collector: %v", err)
	}

	tearDown := func() {
		tCtx.Cancel("tearing down")
		result.TearDownFn()
	}
	syncPeriod := 5 * time.Second
	startGC := func(workers int) {
		go wait.Until(func() {
			// Resetting the REST mapper will also invalidate the underlying discovery
			// client. This is a leaky abstraction and assumes behavior about the REST
			// mapper, but we'll deal with it for now.
			restMapper.Reset()
		}, syncPeriod, tCtx.Done())
		go gc.Run(tCtx, workers, syncPeriod)
		go gc.Sync(tCtx, clientSet.Discovery(), syncPeriod)
	}

	if workerCount > 0 {
		startGC(workerCount)
	}

	return &testContext{
		logger:             logger,
		tearDown:           tearDown,
		gc:                 gc,
		clientSet:          clientSet,
		apiExtensionClient: apiExtensionClient,
		dynamicClient:      dynamicClient,
		metadataClient:     metadataClient,
		startGC:            startGC,
		syncPeriod:         syncPeriod,
	}
}

func createNamespaceOrDie(name string, c clientset.Interface, t *testing.T) *v1.Namespace {
	ns := &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: name}}
	if _, err := c.CoreV1().Namespaces().Create(context.TODO(), ns, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create namespace: %v", err)
	}
	falseVar := false
	_, err := c.CoreV1().ServiceAccounts(ns.Name).Create(context.TODO(), &v1.ServiceAccount{
		ObjectMeta:                   metav1.ObjectMeta{Name: "default"},
		AutomountServiceAccountToken: &falseVar,
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create service account: %v", err)
	}
	return ns
}

func deleteNamespaceOrDie(name string, c clientset.Interface, t *testing.T) {
	zero := int64(0)
	background := metav1.DeletePropagationBackground
	err := c.CoreV1().Namespaces().Delete(context.TODO(), name, metav1.DeleteOptions{GracePeriodSeconds: &zero, PropagationPolicy: &background})
	if err != nil {
		t.Fatalf("failed to delete namespace %q: %v", name, err)
	}
}

func TestCrossNamespaceReferencesWithWatchCache(t *testing.T) {
	testCrossNamespaceReferences(t, true)
}
func TestCrossNamespaceReferencesWithoutWatchCache(t *testing.T) {
	testCrossNamespaceReferences(t, false)
}

func testCrossNamespaceReferences(t *testing.T, watchCache bool) {
	var (
		workers            = 5
		validChildrenCount = 10
		namespaceB         = "b"
		namespaceA         = "a"
	)

	// Start the server
	testServer := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{fmt.Sprintf("--watch-cache=%v", watchCache)}, framework.SharedEtcd())
	defer func() {
		if testServer != nil {
			testServer.TearDownFn()
		}
	}()
	clientSet, err := clientset.NewForConfig(testServer.ClientConfig)
	if err != nil {
		t.Fatalf("error creating clientset: %v", err)
	}

	createNamespaceOrDie(namespaceB, clientSet, t)
	parent, err := clientSet.CoreV1().ConfigMaps(namespaceB).Create(context.TODO(), &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "parent"}}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < validChildrenCount; i++ {
		_, err := clientSet.CoreV1().Secrets(namespaceB).Create(context.TODO(), &v1.Secret{ObjectMeta: metav1.ObjectMeta{GenerateName: "child-", OwnerReferences: []metav1.OwnerReference{
			{Name: "parent", Kind: "ConfigMap", APIVersion: "v1", UID: parent.UID, Controller: ptr.To(false)},
		}}}, metav1.CreateOptions{})
		if err != nil {
			t.Fatal(err)
		}
	}

	createNamespaceOrDie(namespaceA, clientSet, t)

	// Construct invalid owner references:
	invalidOwnerReferences := []metav1.OwnerReference{}
	for i := 0; i < 25; i++ {
		invalidOwnerReferences = append(invalidOwnerReferences, metav1.OwnerReference{Name: "invalid", UID: types.UID(fmt.Sprintf("invalid-%d", i)), APIVersion: "test/v1", Kind: fmt.Sprintf("invalid%d", i)})
	}
	invalidOwnerReferences = append(invalidOwnerReferences, metav1.OwnerReference{Name: "invalid", UID: parent.UID, APIVersion: "v1", Kind: "Pod", Controller: ptr.To(false)})

	for i := 0; i < workers; i++ {
		_, err := clientSet.CoreV1().ConfigMaps(namespaceA).Create(context.TODO(), &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{GenerateName: "invalid-child-", OwnerReferences: invalidOwnerReferences}}, metav1.CreateOptions{})
		if err != nil {
			t.Fatal(err)
		}
		_, err = clientSet.CoreV1().Secrets(namespaceA).Create(context.TODO(), &v1.Secret{ObjectMeta: metav1.ObjectMeta{GenerateName: "invalid-child-a-", OwnerReferences: invalidOwnerReferences}}, metav1.CreateOptions{})
		if err != nil {
			t.Fatal(err)
		}
		_, err = clientSet.CoreV1().Secrets(namespaceA).Create(context.TODO(), &v1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Labels:          map[string]string{"single-bad-reference": "true"},
				GenerateName:    "invalid-child-b-",
				OwnerReferences: []metav1.OwnerReference{{Name: "invalid", UID: parent.UID, APIVersion: "v1", Kind: "Pod", Controller: ptr.To(false)}},
			},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatal(err)
		}
	}

	// start GC with existing objects in place to simulate controller-manager restart
	ctx := setupWithServer(t, testServer, workers)
	defer ctx.tearDown()
	testServer = nil

	// Wait for the invalid children to be garbage collected
	if err := wait.PollImmediate(time.Second, 10*time.Second, func() (bool, error) {
		children, err := clientSet.CoreV1().Secrets(namespaceA).List(context.TODO(), metav1.ListOptions{LabelSelector: "single-bad-reference=true"})
		if err != nil {
			return false, err
		}
		if len(children.Items) > 0 {
			t.Logf("expected 0 invalid children, got %d, will wait and relist", len(children.Items))
			return false, nil
		}
		return true, nil
	}); err != nil && err != wait.ErrWaitTimeout {
		t.Error(err)
	}

	// Wait for a little while to make sure they didn't trigger deletion of the valid children
	if err := wait.Poll(time.Second, 5*time.Second, func() (bool, error) {
		children, err := clientSet.CoreV1().Secrets(namespaceB).List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		if len(children.Items) != validChildrenCount {
			return false, fmt.Errorf("expected %d valid children, got %d", validChildrenCount, len(children.Items))
		}
		return false, nil
	}); err != nil && err != wait.ErrWaitTimeout {
		t.Error(err)
	}

	if !ctx.gc.GraphHasUID(parent.UID) {
		t.Errorf("valid parent UID no longer exists in the graph")
	}

	// Now that our graph has correct data in it, add a new invalid child and see if it gets deleted
	invalidChild, err := clientSet.CoreV1().Secrets(namespaceA).Create(context.TODO(), &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName:    "invalid-child-c-",
			OwnerReferences: []metav1.OwnerReference{{Name: "invalid", UID: parent.UID, APIVersion: "v1", Kind: "Pod", Controller: ptr.To(false)}},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	// Wait for the invalid child to be garbage collected
	if err := wait.PollImmediate(time.Second, 10*time.Second, func() (bool, error) {
		_, err := clientSet.CoreV1().Secrets(namespaceA).Get(context.TODO(), invalidChild.Name, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		if err != nil {
			return false, err
		}
		t.Logf("%s remains, waiting for deletion", invalidChild.Name)
		return false, nil
	}); err != nil {
		t.Fatal(err)
	}
}

// This test simulates the cascading deletion.
func TestCascadingDeletion(t *testing.T) {
	ctx := setup(t, 5)
	defer ctx.tearDown()

	gc, clientSet := ctx.gc, ctx.clientSet

	ns := createNamespaceOrDie("gc-cascading-deletion", clientSet, t)
	defer deleteNamespaceOrDie(ns.Name, clientSet, t)

	rcClient := clientSet.CoreV1().ReplicationControllers(ns.Name)
	podClient := clientSet.CoreV1().Pods(ns.Name)

	toBeDeletedRC, err := rcClient.Create(context.TODO(), newOwnerRC(toBeDeletedRCName, ns.Name), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create replication controller: %v", err)
	}
	remainingRC, err := rcClient.Create(context.TODO(), newOwnerRC(remainingRCName, ns.Name), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create replication controller: %v", err)
	}

	rcs, err := rcClient.List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list replication controllers: %v", err)
	}
	if len(rcs.Items) != 2 {
		t.Fatalf("Expect only 2 replication controller")
	}

	// this pod should be cascadingly deleted.
	pod := newPod(garbageCollectedPodName, ns.Name, []metav1.OwnerReference{{UID: toBeDeletedRC.ObjectMeta.UID, Name: toBeDeletedRCName}})
	_, err = podClient.Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}

	// this pod shouldn't be cascadingly deleted, because it has a valid reference.
	pod = newPod(oneValidOwnerPodName, ns.Name, []metav1.OwnerReference{
		{UID: toBeDeletedRC.ObjectMeta.UID, Name: toBeDeletedRCName},
		{UID: remainingRC.ObjectMeta.UID, Name: remainingRCName},
	})
	_, err = podClient.Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}

	// this pod shouldn't be cascadingly deleted, because it doesn't have an owner.
	pod = newPod(independentPodName, ns.Name, []metav1.OwnerReference{})
	_, err = podClient.Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}

	// set up watch
	pods, err := podClient.List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list pods: %v", err)
	}
	if len(pods.Items) != 3 {
		t.Fatalf("Expect only 3 pods")
	}
	// delete one of the replication controller
	if err := rcClient.Delete(context.TODO(), toBeDeletedRCName, getNonOrphanOptions()); err != nil {
		t.Fatalf("failed to delete replication controller: %v", err)
	}
	// sometimes the deletion of the RC takes long time to be observed by
	// the gc, so wait for the garbage collector to observe the deletion of
	// the toBeDeletedRC
	if err := wait.Poll(1*time.Second, 60*time.Second, func() (bool, error) {
		return !gc.GraphHasUID(toBeDeletedRC.ObjectMeta.UID), nil
	}); err != nil {
		t.Fatal(err)
	}
	if err := integration.WaitForPodToDisappear(podClient, garbageCollectedPodName, 1*time.Second, 30*time.Second); err != nil {
		t.Fatalf("expect pod %s to be garbage collected, got err= %v", garbageCollectedPodName, err)
	}
	// checks the garbage collect doesn't delete pods it shouldn't delete.
	if _, err := podClient.Get(context.TODO(), independentPodName, metav1.GetOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, err := podClient.Get(context.TODO(), oneValidOwnerPodName, metav1.GetOptions{}); err != nil {
		t.Fatal(err)
	}
}

// This test simulates the case where an object is created with an owner that
// doesn't exist. It verifies the GC will delete such an object.
func TestCreateWithNonExistentOwner(t *testing.T) {
	ctx := setup(t, 5)
	defer ctx.tearDown()

	clientSet := ctx.clientSet

	ns := createNamespaceOrDie("gc-non-existing-owner", clientSet, t)
	defer deleteNamespaceOrDie(ns.Name, clientSet, t)

	podClient := clientSet.CoreV1().Pods(ns.Name)

	pod := newPod(garbageCollectedPodName, ns.Name, []metav1.OwnerReference{{UID: "doesn't matter", Name: toBeDeletedRCName}})
	_, err := podClient.Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}

	// set up watch
	pods, err := podClient.List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list pods: %v", err)
	}
	if len(pods.Items) > 1 {
		t.Fatalf("Unexpected pod list: %v", pods.Items)
	}
	// wait for the garbage collector to delete the pod
	if err := integration.WaitForPodToDisappear(podClient, garbageCollectedPodName, 1*time.Second, 30*time.Second); err != nil {
		t.Fatalf("expect pod %s to be garbage collected, got err= %v", garbageCollectedPodName, err)
	}
}

func setupRCsPods(t *testing.T, gc *garbagecollector.GarbageCollector, clientSet clientset.Interface, nameSuffix, namespace string, initialFinalizers []string, options metav1.DeleteOptions, wg *sync.WaitGroup, rcUIDs chan types.UID, errs chan string) {
	defer wg.Done()
	rcClient := clientSet.CoreV1().ReplicationControllers(namespace)
	podClient := clientSet.CoreV1().Pods(namespace)
	// create rc.
	rcName := "test.rc." + nameSuffix
	rc := newOwnerRC(rcName, namespace)
	rc.ObjectMeta.Finalizers = initialFinalizers
	rc, err := rcClient.Create(context.TODO(), rc, metav1.CreateOptions{})
	if err != nil {
		errs <- fmt.Sprintf("Failed to create replication controller: %v", err)
		return
	}
	rcUIDs <- rc.ObjectMeta.UID
	// create pods.
	var podUIDs []types.UID
	for j := 0; j < 3; j++ {
		podName := "test.pod." + nameSuffix + "-" + strconv.Itoa(j)
		pod := newPod(podName, namespace, []metav1.OwnerReference{{UID: rc.ObjectMeta.UID, Name: rc.ObjectMeta.Name}})
		createdPod, err := podClient.Create(context.TODO(), pod, metav1.CreateOptions{})
		if err != nil {
			errs <- fmt.Sprintf("Failed to create Pod: %v", err)
			return
		}
		podUIDs = append(podUIDs, createdPod.ObjectMeta.UID)
	}
	orphan := false
	switch {
	case options.OrphanDependents == nil && options.PropagationPolicy == nil && len(initialFinalizers) == 0: //nolint:staticcheck // SA1019 Keep testing deprecated OrphanDependents option until it's being removed
		// if there are no deletion options, the default policy for replication controllers is orphan
		orphan = true
	case options.OrphanDependents != nil: //nolint:staticcheck // SA1019 Keep testing deprecated OrphanDependents option until it's being removed
		// if the deletion options explicitly specify whether to orphan, that controls
		orphan = *options.OrphanDependents //nolint:staticcheck // SA1019 Keep testing deprecated OrphanDependents option until it's being removed
	case options.PropagationPolicy != nil:
		// if the deletion options explicitly specify whether to orphan, that controls
		orphan = *options.PropagationPolicy == metav1.DeletePropagationOrphan
	case len(initialFinalizers) != 0 && initialFinalizers[0] == metav1.FinalizerOrphanDependents:
		// if the orphan finalizer is explicitly added, we orphan
		orphan = true
	}
	// if we intend to orphan the pods, we need wait for the gc to observe the
	// creation of the pods, otherwise if the deletion of RC is observed before
	// the creation of the pods, the pods will not be orphaned.
	if orphan {
		err := wait.Poll(1*time.Second, 60*time.Second, func() (bool, error) {
			for _, u := range podUIDs {
				if !gc.GraphHasUID(u) {
					return false, nil
				}
			}
			return true, nil
		})
		if err != nil {
			errs <- fmt.Sprintf("failed to observe the expected pods in the GC graph for rc %s", rcName)
			return
		}
	}
	// delete the rc
	if err := rcClient.Delete(context.TODO(), rc.ObjectMeta.Name, options); err != nil {
		errs <- fmt.Sprintf("failed to delete replication controller: %v", err)
		return
	}
}

func verifyRemainingObjects(t *testing.T, clientSet clientset.Interface, namespace string, rcNum, podNum int) (bool, error) {
	rcClient := clientSet.CoreV1().ReplicationControllers(namespace)
	podClient := clientSet.CoreV1().Pods(namespace)
	pods, err := podClient.List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("Failed to list pods: %v", err)
	}
	var ret = true
	if len(pods.Items) != podNum {
		ret = false
		t.Logf("expect %d pods, got %d pods", podNum, len(pods.Items))
	}
	rcs, err := rcClient.List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("Failed to list replication controllers: %v", err)
	}
	if len(rcs.Items) != rcNum {
		ret = false
		t.Logf("expect %d RCs, got %d RCs", rcNum, len(rcs.Items))
	}
	return ret, nil
}

// The stress test is not very stressful, because we need to control the running
// time of our pre-submit tests to increase submit-queue throughput. We'll add
// e2e tests that put more stress.
func TestStressingCascadingDeletion(t *testing.T) {
	ctx := setup(t, 5)
	defer ctx.tearDown()

	gc, clientSet := ctx.gc, ctx.clientSet

	ns := createNamespaceOrDie("gc-stressing-cascading-deletion", clientSet, t)
	defer deleteNamespaceOrDie(ns.Name, clientSet, t)

	const collections = 10
	var wg sync.WaitGroup
	wg.Add(collections * 5)
	rcUIDs := make(chan types.UID, collections*5)
	errs := make(chan string, 5)
	for i := 0; i < collections; i++ {
		// rc is created with empty finalizers, deleted with nil delete options, pods will remain.
		go setupRCsPods(t, gc, clientSet, "collection1-"+strconv.Itoa(i), ns.Name, []string{}, metav1.DeleteOptions{}, &wg, rcUIDs, errs)
		// rc is created with the orphan finalizer, deleted with nil options, pods will remain.
		go setupRCsPods(t, gc, clientSet, "collection2-"+strconv.Itoa(i), ns.Name, []string{metav1.FinalizerOrphanDependents}, metav1.DeleteOptions{}, &wg, rcUIDs, errs)
		// rc is created with the orphan finalizer, deleted with DeleteOptions.OrphanDependents=false, pods will be deleted.
		go setupRCsPods(t, gc, clientSet, "collection3-"+strconv.Itoa(i), ns.Name, []string{metav1.FinalizerOrphanDependents}, getNonOrphanOptions(), &wg, rcUIDs, errs)
		// rc is created with empty finalizers, deleted with DeleteOptions.OrphanDependents=true, pods will remain.
		go setupRCsPods(t, gc, clientSet, "collection4-"+strconv.Itoa(i), ns.Name, []string{}, getOrphanOptions(), &wg, rcUIDs, errs)
		// rc is created with empty finalizers, deleted with DeleteOptions.PropagationPolicy=Orphan, pods will remain.
		go setupRCsPods(t, gc, clientSet, "collection5-"+strconv.Itoa(i), ns.Name, []string{}, getPropagateOrphanOptions(), &wg, rcUIDs, errs)
	}
	wg.Wait()
	close(errs)
	for errString := range errs {
		t.Fatal(errString)
	}
	t.Logf("all pods are created, all replications controllers are created then deleted")
	// wait for the RCs and Pods to reach the expected numbers.
	if err := wait.Poll(1*time.Second, 300*time.Second, func() (bool, error) {
		podsInEachCollection := 3
		// see the comments on the calls to setupRCsPods for details
		remainingGroups := 4
		return verifyRemainingObjects(t, clientSet, ns.Name, 0, collections*podsInEachCollection*remainingGroups)
	}); err != nil {
		t.Fatal(err)
	}
	t.Logf("number of remaining replication controllers and pods are as expected")

	// verify the remaining pods all have "orphan" in their names.
	podClient := clientSet.CoreV1().Pods(ns.Name)
	pods, err := podClient.List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for _, pod := range pods.Items {
		if !strings.Contains(pod.ObjectMeta.Name, "collection1-") && !strings.Contains(pod.ObjectMeta.Name, "collection2-") && !strings.Contains(pod.ObjectMeta.Name, "collection4-") && !strings.Contains(pod.ObjectMeta.Name, "collection5-") {
			t.Errorf("got unexpected remaining pod: %#v", pod)
		}
	}

	// verify there is no node representing replication controllers in the gc's graph
	for i := 0; i < collections; i++ {
		uid := <-rcUIDs
		if gc.GraphHasUID(uid) {
			t.Errorf("Expect all nodes representing replication controllers are removed from the Propagator's graph")
		}
	}
}

func TestOrphaning(t *testing.T) {
	ctx := setup(t, 5)
	defer ctx.tearDown()

	gc, clientSet := ctx.gc, ctx.clientSet

	ns := createNamespaceOrDie("gc-orphaning", clientSet, t)
	defer deleteNamespaceOrDie(ns.Name, clientSet, t)

	podClient := clientSet.CoreV1().Pods(ns.Name)
	rcClient := clientSet.CoreV1().ReplicationControllers(ns.Name)
	// create the RC with the orphan finalizer set
	toBeDeletedRC := newOwnerRC(toBeDeletedRCName, ns.Name)
	toBeDeletedRC, err := rcClient.Create(context.TODO(), toBeDeletedRC, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create replication controller: %v", err)
	}

	// these pods should be orphaned.
	var podUIDs []types.UID
	podsNum := 3
	for i := 0; i < podsNum; i++ {
		podName := garbageCollectedPodName + strconv.Itoa(i)
		pod := newPod(podName, ns.Name, []metav1.OwnerReference{{UID: toBeDeletedRC.ObjectMeta.UID, Name: toBeDeletedRCName}})
		createdPod, err := podClient.Create(context.TODO(), pod, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create Pod: %v", err)
		}
		podUIDs = append(podUIDs, createdPod.ObjectMeta.UID)
	}

	// we need wait for the gc to observe the creation of the pods, otherwise if
	// the deletion of RC is observed before the creation of the pods, the pods
	// will not be orphaned.
	err = wait.Poll(1*time.Second, 60*time.Second, func() (bool, error) {
		for _, u := range podUIDs {
			if !gc.GraphHasUID(u) {
				return false, nil
			}
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Failed to observe pods in GC graph for %s: %v", toBeDeletedRC.Name, err)
	}

	err = rcClient.Delete(context.TODO(), toBeDeletedRCName, getOrphanOptions())
	if err != nil {
		t.Fatalf("Failed to gracefully delete the rc: %v", err)
	}
	// verify the toBeDeleteRC is deleted
	if err := wait.PollImmediate(1*time.Second, 30*time.Second, func() (bool, error) {
		rcs, err := rcClient.List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		if len(rcs.Items) == 0 {
			t.Logf("Still has %d RCs", len(rcs.Items))
			return true, nil
		}
		return false, nil
	}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// verify pods don't have the ownerPod as an owner anymore
	pods, err := podClient.List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list pods: %v", err)
	}
	if len(pods.Items) != podsNum {
		t.Errorf("Expect %d pod(s), but got %#v", podsNum, pods)
	}
	for _, pod := range pods.Items {
		if len(pod.ObjectMeta.OwnerReferences) != 0 {
			t.Errorf("pod %s still has non-empty OwnerReferences: %v", pod.ObjectMeta.Name, pod.ObjectMeta.OwnerReferences)
		}
	}
}

func TestSolidOwnerDoesNotBlockWaitingOwner(t *testing.T) {
	ctx := setup(t, 5)
	defer ctx.tearDown()

	clientSet := ctx.clientSet

	ns := createNamespaceOrDie("gc-foreground1", clientSet, t)
	defer deleteNamespaceOrDie(ns.Name, clientSet, t)

	podClient := clientSet.CoreV1().Pods(ns.Name)
	rcClient := clientSet.CoreV1().ReplicationControllers(ns.Name)
	// create the RC with the orphan finalizer set
	toBeDeletedRC, err := rcClient.Create(context.TODO(), newOwnerRC(toBeDeletedRCName, ns.Name), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create replication controller: %v", err)
	}
	remainingRC, err := rcClient.Create(context.TODO(), newOwnerRC(remainingRCName, ns.Name), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create replication controller: %v", err)
	}
	trueVar := true
	pod := newPod("pod", ns.Name, []metav1.OwnerReference{
		{UID: toBeDeletedRC.ObjectMeta.UID, Name: toBeDeletedRC.Name, BlockOwnerDeletion: &trueVar},
		{UID: remainingRC.ObjectMeta.UID, Name: remainingRC.Name},
	})
	_, err = podClient.Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}

	err = rcClient.Delete(context.TODO(), toBeDeletedRCName, getForegroundOptions())
	if err != nil {
		t.Fatalf("Failed to delete the rc: %v", err)
	}
	// verify the toBeDeleteRC is deleted
	if err := wait.PollImmediate(1*time.Second, 30*time.Second, func() (bool, error) {
		_, err := rcClient.Get(context.TODO(), toBeDeletedRC.Name, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				return true, nil
			}
			return false, err
		}
		return false, nil
	}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// verify pods don't have the toBeDeleteRC as an owner anymore
	pod, err = podClient.Get(context.TODO(), "pod", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to list pods: %v", err)
	}
	if len(pod.ObjectMeta.OwnerReferences) != 1 {
		t.Errorf("expect pod to have only one ownerReference: got %#v", pod.ObjectMeta.OwnerReferences)
	} else if pod.ObjectMeta.OwnerReferences[0].Name != remainingRC.Name {
		t.Errorf("expect pod to have an ownerReference pointing to %s, got %#v", remainingRC.Name, pod.ObjectMeta.OwnerReferences)
	}
}

func TestNonBlockingOwnerRefDoesNotBlock(t *testing.T) {
	ctx := setup(t, 5)
	defer ctx.tearDown()

	clientSet := ctx.clientSet

	ns := createNamespaceOrDie("gc-foreground2", clientSet, t)
	defer deleteNamespaceOrDie(ns.Name, clientSet, t)

	podClient := clientSet.CoreV1().Pods(ns.Name)
	rcClient := clientSet.CoreV1().ReplicationControllers(ns.Name)
	// create the RC with the orphan finalizer set
	toBeDeletedRC, err := rcClient.Create(context.TODO(), newOwnerRC(toBeDeletedRCName, ns.Name), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create replication controller: %v", err)
	}
	// BlockingOwnerDeletion is not set
	pod1 := newPod("pod1", ns.Name, []metav1.OwnerReference{
		{UID: toBeDeletedRC.ObjectMeta.UID, Name: toBeDeletedRC.Name},
	})
	// adding finalizer that no controller handles, so that the pod won't be deleted
	pod1.ObjectMeta.Finalizers = []string{"x/y"}
	// BlockingOwnerDeletion is false
	falseVar := false
	pod2 := newPod("pod2", ns.Name, []metav1.OwnerReference{
		{UID: toBeDeletedRC.ObjectMeta.UID, Name: toBeDeletedRC.Name, BlockOwnerDeletion: &falseVar},
	})
	// adding finalizer that no controller handles, so that the pod won't be deleted
	pod2.ObjectMeta.Finalizers = []string{"x/y"}
	_, err = podClient.Create(context.TODO(), pod1, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}
	_, err = podClient.Create(context.TODO(), pod2, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}

	err = rcClient.Delete(context.TODO(), toBeDeletedRCName, getForegroundOptions())
	if err != nil {
		t.Fatalf("Failed to delete the rc: %v", err)
	}
	// verify the toBeDeleteRC is deleted
	if err := wait.PollImmediate(1*time.Second, 30*time.Second, func() (bool, error) {
		_, err := rcClient.Get(context.TODO(), toBeDeletedRC.Name, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				return true, nil
			}
			return false, err
		}
		return false, nil
	}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// verify pods are still there
	pods, err := podClient.List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list pods: %v", err)
	}
	if len(pods.Items) != 2 {
		t.Errorf("expect there to be 2 pods, got %#v", pods.Items)
	}
}

func TestDoubleDeletionWithFinalizer(t *testing.T) {
	// test setup
	ctx := setup(t, 5)
	defer ctx.tearDown()
	clientSet := ctx.clientSet
	ns := createNamespaceOrDie("gc-double-foreground", clientSet, t)
	defer deleteNamespaceOrDie(ns.Name, clientSet, t)

	// step 1: creates a pod with a custom finalizer and deletes it, then waits until gc removes its finalizer
	podClient := clientSet.CoreV1().Pods(ns.Name)
	pod := newPod("lucy", ns.Name, nil)
	pod.ObjectMeta.Finalizers = []string{"x/y"}
	if _, err := podClient.Create(context.TODO(), pod, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}
	if err := podClient.Delete(context.TODO(), pod.Name, getForegroundOptions()); err != nil {
		t.Fatalf("Failed to delete pod: %v", err)
	}
	if err := wait.PollImmediate(1*time.Second, 10*time.Second, func() (bool, error) {
		returnedPod, err := podClient.Get(context.TODO(), pod.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if len(returnedPod.Finalizers) != 1 || returnedPod.Finalizers[0] != "x/y" {
			t.Logf("waiting for pod %q to have only one finalizer %q at step 1, got %v", returnedPod.Name, "x/y", returnedPod.Finalizers)
			return false, nil
		}
		return true, nil
	}); err != nil {
		t.Fatalf("Failed waiting for pod to have only one filanizer at step 1, error: %v", err)
	}

	// step 2: deletes the pod one more time and checks if there's only the custom finalizer left
	if err := podClient.Delete(context.TODO(), pod.Name, getForegroundOptions()); err != nil {
		t.Fatalf("Failed to delete pod: %v", err)
	}
	if err := wait.PollImmediate(1*time.Second, 10*time.Second, func() (bool, error) {
		returnedPod, err := podClient.Get(context.TODO(), pod.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if len(returnedPod.Finalizers) != 1 || returnedPod.Finalizers[0] != "x/y" {
			t.Logf("waiting for pod %q to have only one finalizer %q at step 2, got %v", returnedPod.Name, "x/y", returnedPod.Finalizers)
			return false, nil
		}
		return true, nil
	}); err != nil {
		t.Fatalf("Failed waiting for pod to have only one finalizer at step 2, gc hasn't removed its finalzier?, error: %v", err)
	}

	// step 3: removes the custom finalizer and checks if the pod was removed
	patch := []byte(`[{"op":"remove","path":"/metadata/finalizers"}]`)
	if _, err := podClient.Patch(context.TODO(), pod.Name, types.JSONPatchType, patch, metav1.PatchOptions{}); err != nil {
		t.Fatalf("Failed to update pod: %v", err)
	}
	if err := wait.Poll(1*time.Second, 10*time.Second, func() (bool, error) {
		_, err := podClient.Get(context.TODO(), pod.Name, metav1.GetOptions{})
		return apierrors.IsNotFound(err), nil
	}); err != nil {
		t.Fatalf("Failed waiting for pod %q to be deleted", pod.Name)
	}
}

func TestBlockingOwnerRefDoesBlock(t *testing.T) {
	ctx := setup(t, 0)
	defer ctx.tearDown()
	gc, clientSet := ctx.gc, ctx.clientSet

	ns := createNamespaceOrDie("foo", clientSet, t)
	defer deleteNamespaceOrDie(ns.Name, clientSet, t)

	podClient := clientSet.CoreV1().Pods(ns.Name)
	rcClient := clientSet.CoreV1().ReplicationControllers(ns.Name)
	// create the RC with the orphan finalizer set
	toBeDeletedRC, err := rcClient.Create(context.TODO(), newOwnerRC(toBeDeletedRCName, ns.Name), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create replication controller: %v", err)
	}
	trueVar := true
	pod := newPod("pod", ns.Name, []metav1.OwnerReference{
		{UID: toBeDeletedRC.ObjectMeta.UID, Name: toBeDeletedRC.Name, BlockOwnerDeletion: &trueVar},
	})
	// adding finalizer that no controller handles, so that the pod won't be deleted
	pod.ObjectMeta.Finalizers = []string{"x/y"}
	_, err = podClient.Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}

	// this makes sure the garbage collector will have added the pod to its
	// dependency graph before handling the foreground deletion of the rc.
	ctx.startGC(5)
	timeout := make(chan struct{})
	time.AfterFunc(5*time.Second, func() { close(timeout) })
	if !cache.WaitForCacheSync(timeout, func() bool {
		return gc.IsSynced(ctx.logger)
	}) {
		t.Fatalf("failed to wait for garbage collector to be synced")
	}

	err = rcClient.Delete(context.TODO(), toBeDeletedRCName, getForegroundOptions())
	if err != nil {
		t.Fatalf("Failed to delete the rc: %v", err)
	}
	time.Sleep(15 * time.Second)
	// verify the toBeDeleteRC is NOT deleted
	_, err = rcClient.Get(context.TODO(), toBeDeletedRC.Name, metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// verify pods are still there
	pods, err := podClient.List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list pods: %v", err)
	}
	if len(pods.Items) != 1 {
		t.Errorf("expect there to be 1 pods, got %#v", pods.Items)
	}
}

// TestCustomResourceCascadingDeletion ensures the basic cascading delete
// behavior supports custom resources.
func TestCustomResourceCascadingDeletion(t *testing.T) {
	ctx := setup(t, 5)
	defer ctx.tearDown()

	clientSet, apiExtensionClient, dynamicClient := ctx.clientSet, ctx.apiExtensionClient, ctx.dynamicClient

	ns := createNamespaceOrDie("crd-cascading", clientSet, t)

	definition, resourceClient := createRandomCustomResourceDefinition(t, apiExtensionClient, dynamicClient, ns.Name)

	// Create a custom owner resource.
	owner := newCRDInstance(definition, ns.Name, names.SimpleNameGenerator.GenerateName("owner"))
	owner, err := resourceClient.Create(context.TODO(), owner, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create owner resource %q: %v", owner.GetName(), err)
	}
	t.Logf("created owner resource %q", owner.GetName())

	// Create a custom dependent resource.
	dependent := newCRDInstance(definition, ns.Name, names.SimpleNameGenerator.GenerateName("dependent"))
	link(t, owner, dependent)

	dependent, err = resourceClient.Create(context.TODO(), dependent, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create dependent resource %q: %v", dependent.GetName(), err)
	}
	t.Logf("created dependent resource %q", dependent.GetName())

	// Delete the owner.
	foreground := metav1.DeletePropagationForeground
	err = resourceClient.Delete(context.TODO(), owner.GetName(), metav1.DeleteOptions{PropagationPolicy: &foreground})
	if err != nil {
		t.Fatalf("failed to delete owner resource %q: %v", owner.GetName(), err)
	}

	// Ensure the owner is deleted.
	if err := wait.Poll(1*time.Second, 60*time.Second, func() (bool, error) {
		_, err := resourceClient.Get(context.TODO(), owner.GetName(), metav1.GetOptions{})
		return apierrors.IsNotFound(err), nil
	}); err != nil {
		t.Fatalf("failed waiting for owner resource %q to be deleted", owner.GetName())
	}

	// Ensure the dependent is deleted.
	_, err = resourceClient.Get(context.TODO(), dependent.GetName(), metav1.GetOptions{})
	if err == nil {
		t.Fatalf("expected dependent %q to be deleted", dependent.GetName())
	} else {
		if !apierrors.IsNotFound(err) {
			t.Fatalf("unexpected error getting dependent %q: %v", dependent.GetName(), err)
		}
	}
}

// TestMixedRelationships ensures that owner/dependent relationships work
// between core and custom resources.
//
// TODO: Consider how this could be represented with table-style tests (e.g. a
// before/after expected object graph given a delete operation targeting a
// specific node in the before graph with certain delete options).
func TestMixedRelationships(t *testing.T) {
	ctx := setup(t, 5)
	defer ctx.tearDown()

	clientSet, apiExtensionClient, dynamicClient := ctx.clientSet, ctx.apiExtensionClient, ctx.dynamicClient

	ns := createNamespaceOrDie("crd-mixed", clientSet, t)

	configMapClient := clientSet.CoreV1().ConfigMaps(ns.Name)

	definition, resourceClient := createRandomCustomResourceDefinition(t, apiExtensionClient, dynamicClient, ns.Name)

	// Create a custom owner resource.
	customOwner, err := resourceClient.Create(context.TODO(), newCRDInstance(definition, ns.Name, names.SimpleNameGenerator.GenerateName("owner")), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create owner: %v", err)
	}
	t.Logf("created custom owner %q", customOwner.GetName())

	// Create a core dependent resource.
	coreDependent := newConfigMap(ns.Name, names.SimpleNameGenerator.GenerateName("dependent"))
	link(t, customOwner, coreDependent)
	coreDependent, err = configMapClient.Create(context.TODO(), coreDependent, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create dependent: %v", err)
	}
	t.Logf("created core dependent %q", coreDependent.GetName())

	// Create a core owner resource.
	coreOwner, err := configMapClient.Create(context.TODO(), newConfigMap(ns.Name, names.SimpleNameGenerator.GenerateName("owner")), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create owner: %v", err)
	}
	t.Logf("created core owner %q: %#v", coreOwner.GetName(), coreOwner)

	// Create a custom dependent resource.
	customDependent := newCRDInstance(definition, ns.Name, names.SimpleNameGenerator.GenerateName("dependent"))
	coreOwner.TypeMeta.Kind = "ConfigMap"
	coreOwner.TypeMeta.APIVersion = "v1"
	link(t, coreOwner, customDependent)
	customDependent, err = resourceClient.Create(context.TODO(), customDependent, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create dependent: %v", err)
	}
	t.Logf("created custom dependent %q", customDependent.GetName())

	// Delete the custom owner.
	foreground := metav1.DeletePropagationForeground
	err = resourceClient.Delete(context.TODO(), customOwner.GetName(), metav1.DeleteOptions{PropagationPolicy: &foreground})
	if err != nil {
		t.Fatalf("failed to delete owner resource %q: %v", customOwner.GetName(), err)
	}

	// Ensure the owner is deleted.
	if err := wait.Poll(1*time.Second, 60*time.Second, func() (bool, error) {
		_, err := resourceClient.Get(context.TODO(), customOwner.GetName(), metav1.GetOptions{})
		return apierrors.IsNotFound(err), nil
	}); err != nil {
		t.Fatalf("failed waiting for owner resource %q to be deleted", customOwner.GetName())
	}

	// Ensure the dependent is deleted.
	_, err = resourceClient.Get(context.TODO(), coreDependent.GetName(), metav1.GetOptions{})
	if err == nil {
		t.Fatalf("expected dependent %q to be deleted", coreDependent.GetName())
	} else {
		if !apierrors.IsNotFound(err) {
			t.Fatalf("unexpected error getting dependent %q: %v", coreDependent.GetName(), err)
		}
	}

	// Delete the core owner.
	err = configMapClient.Delete(context.TODO(), coreOwner.GetName(), metav1.DeleteOptions{PropagationPolicy: &foreground})
	if err != nil {
		t.Fatalf("failed to delete owner resource %q: %v", coreOwner.GetName(), err)
	}

	// Ensure the owner is deleted.
	if err := wait.Poll(1*time.Second, 60*time.Second, func() (bool, error) {
		_, err := configMapClient.Get(context.TODO(), coreOwner.GetName(), metav1.GetOptions{})
		return apierrors.IsNotFound(err), nil
	}); err != nil {
		t.Fatalf("failed waiting for owner resource %q to be deleted", coreOwner.GetName())
	}

	// Ensure the dependent is deleted.
	_, err = resourceClient.Get(context.TODO(), customDependent.GetName(), metav1.GetOptions{})
	if err == nil {
		t.Fatalf("expected dependent %q to be deleted", customDependent.GetName())
	} else {
		if !apierrors.IsNotFound(err) {
			t.Fatalf("unexpected error getting dependent %q: %v", customDependent.GetName(), err)
		}
	}
}

// TestCRDDeletionCascading ensures propagating deletion of a custom resource
// definition with an instance that owns a core resource.
func TestCRDDeletionCascading(t *testing.T) {
	ctx := setup(t, 5)
	defer ctx.tearDown()

	clientSet, apiExtensionClient, dynamicClient := ctx.clientSet, ctx.apiExtensionClient, ctx.dynamicClient

	ns := createNamespaceOrDie("crd-mixed", clientSet, t)

	t.Logf("First pass CRD cascading deletion")
	definition, resourceClient := createRandomCustomResourceDefinition(t, apiExtensionClient, dynamicClient, ns.Name)
	testCRDDeletion(t, ctx, ns, definition, resourceClient)

	t.Logf("Second pass CRD cascading deletion")
	accessor := meta.NewAccessor()
	accessor.SetResourceVersion(definition, "")
	_, err := apiextensionstestserver.CreateNewV1CustomResourceDefinition(definition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatalf("failed to create CustomResourceDefinition: %v", err)
	}
	testCRDDeletion(t, ctx, ns, definition, resourceClient)
}

func testCRDDeletion(t *testing.T, ctx *testContext, ns *v1.Namespace, definition *apiextensionsv1.CustomResourceDefinition, resourceClient dynamic.ResourceInterface) {
	clientSet, apiExtensionClient := ctx.clientSet, ctx.apiExtensionClient

	configMapClient := clientSet.CoreV1().ConfigMaps(ns.Name)

	// Create a custom owner resource.
	owner, err := resourceClient.Create(context.TODO(), newCRDInstance(definition, ns.Name, names.SimpleNameGenerator.GenerateName("owner")), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create owner: %v", err)
	}
	t.Logf("created owner %q", owner.GetName())

	// Create a core dependent resource.
	dependent := newConfigMap(ns.Name, names.SimpleNameGenerator.GenerateName("dependent"))
	link(t, owner, dependent)
	dependent, err = configMapClient.Create(context.TODO(), dependent, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create dependent: %v", err)
	}
	t.Logf("created dependent %q", dependent.GetName())

	time.Sleep(ctx.syncPeriod + 5*time.Second)

	// Delete the definition, which should cascade to the owner and ultimately its dependents.
	if err := apiextensionstestserver.DeleteV1CustomResourceDefinition(definition, apiExtensionClient); err != nil {
		t.Fatalf("failed to delete %q: %v", definition.Name, err)
	}

	// Ensure the owner is deleted.
	if err := wait.Poll(1*time.Second, 60*time.Second, func() (bool, error) {
		_, err := resourceClient.Get(context.TODO(), owner.GetName(), metav1.GetOptions{})
		return apierrors.IsNotFound(err), nil
	}); err != nil {
		t.Fatalf("failed waiting for owner %q to be deleted", owner.GetName())
	}

	// Ensure the dependent is deleted.
	if err := wait.Poll(1*time.Second, 60*time.Second, func() (bool, error) {
		_, err := configMapClient.Get(context.TODO(), dependent.GetName(), metav1.GetOptions{})
		return apierrors.IsNotFound(err), nil
	}); err != nil {
		t.Fatalf("failed waiting for dependent %q (owned by %q) to be deleted", dependent.GetName(), owner.GetName())
	}
}

// TestCascadingDeleteOnCRDConversionFailure tests that a bad conversion webhook cannot block the entire GC controller.
// Historically, a cache sync failure from a single resource prevented GC controller from running. This test creates
// a CRD, updates the storage version with a bad conversion webhook and then runs a simple cascading delete test.
func TestCascadingDeleteOnCRDConversionFailure(t *testing.T) {
	ctx := setup(t, 0)
	defer ctx.tearDown()
	gc, apiExtensionClient, dynamicClient, clientSet := ctx.gc, ctx.apiExtensionClient, ctx.dynamicClient, ctx.clientSet

	ns := createNamespaceOrDie("gc-cache-sync-fail", clientSet, t)
	defer deleteNamespaceOrDie(ns.Name, clientSet, t)

	// Create a CRD with storage/serving version v1beta2. Then update the CRD with v1 as the storage version
	// and an invalid conversion webhook. This should result in cache sync failures for the CRD from the GC controller.
	def, dc := createRandomCustomResourceDefinition(t, apiExtensionClient, dynamicClient, ns.Name)
	_, err := dc.Create(context.TODO(), newCRDInstance(def, ns.Name, names.SimpleNameGenerator.GenerateName("test")), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create custom resource: %v", err)
	}

	def, err = apiExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Get(context.TODO(), def.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get custom resource: %v", err)
	}

	newDefinition := def.DeepCopy()
	newDefinition.Spec.Conversion = &apiextensionsv1.CustomResourceConversion{
		Strategy: apiextensionsv1.WebhookConverter,
		Webhook: &apiextensionsv1.WebhookConversion{
			ClientConfig: &apiextensionsv1.WebhookClientConfig{
				Service: &apiextensionsv1.ServiceReference{
					Name:      "foobar",
					Namespace: ns.Name,
				},
				CABundle: testCert,
			},
			ConversionReviewVersions: []string{
				"v1", "v1beta1",
			},
		},
	}
	newDefinition.Spec.Versions = []apiextensionsv1.CustomResourceDefinitionVersion{
		{
			Name:    "v1",
			Served:  true,
			Storage: true,
			Schema:  apiextensionstestserver.AllowAllSchema(),
		},
		{
			Name:    "v1beta1",
			Served:  true,
			Storage: false,
			Schema:  apiextensionstestserver.AllowAllSchema(),
		},
	}

	_, err = apiExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Update(context.TODO(), newDefinition, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Error updating CRD with conversion webhook: %v", err)
	}

	ctx.startGC(5)
	// make sure gc.Sync finds the new CRD and starts monitoring it
	time.Sleep(ctx.syncPeriod + 1*time.Second)

	rcClient := clientSet.CoreV1().ReplicationControllers(ns.Name)
	podClient := clientSet.CoreV1().Pods(ns.Name)

	toBeDeletedRC, err := rcClient.Create(context.TODO(), newOwnerRC(toBeDeletedRCName, ns.Name), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create replication controller: %v", err)
	}

	rcs, err := rcClient.List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list replication controllers: %v", err)
	}
	if len(rcs.Items) != 1 {
		t.Fatalf("Expect only 1 replication controller")
	}

	pod := newPod(garbageCollectedPodName, ns.Name, []metav1.OwnerReference{{UID: toBeDeletedRC.ObjectMeta.UID, Name: toBeDeletedRCName}})
	_, err = podClient.Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}

	pods, err := podClient.List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list pods: %v", err)
	}
	if len(pods.Items) != 1 {
		t.Fatalf("Expect only 1 pods")
	}

	if err := rcClient.Delete(context.TODO(), toBeDeletedRCName, getNonOrphanOptions()); err != nil {
		t.Fatalf("failed to delete replication controller: %v", err)
	}

	// sometimes the deletion of the RC takes long time to be observed by
	// the gc, so wait for the garbage collector to observe the deletion of
	// the toBeDeletedRC
	if err := wait.PollUntilContextTimeout(context.Background(), 1*time.Second, 60*time.Second, true, func(ctx context.Context) (bool, error) {
		return !gc.GraphHasUID(toBeDeletedRC.ObjectMeta.UID), nil
	}); err != nil {
		t.Fatal(err)
	}
	if err := integration.WaitForPodToDisappear(podClient, garbageCollectedPodName, 1*time.Second, 30*time.Second); err != nil {
		t.Fatalf("expect pod %s to be garbage collected, got err= %v", garbageCollectedPodName, err)
	}

	// Check that the cache is still not synced after cascading delete succeeded
	// If this check passes, check that the conversion webhook is correctly misconfigured
	// to prevent watch cache from listing the CRD.
	if ctx.gc.IsSynced(ctx.logger) {
		t.Fatal("cache is not expected to be synced due to bad conversion webhook")
	}
}
