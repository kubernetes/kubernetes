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

package quota

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	apiextensionstestserver "k8s.io/apiextensions-apiserver/test/integration/fixtures"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/admission/plugin/resourcequota"
	resourcequotaapi "k8s.io/apiserver/pkg/admission/plugin/resourcequota/apis/resourcequota"
	"k8s.io/apiserver/pkg/quota/v1/generic"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/metadata"
	"k8s.io/client-go/metadata/metadatainformer"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/record"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/controller-manager/pkg/informerfactory"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller"
	replicationcontroller "k8s.io/kubernetes/pkg/controller/replication"
	resourcequotacontroller "k8s.io/kubernetes/pkg/controller/resourcequota"
	quotainstall "k8s.io/kubernetes/pkg/quota/v1/install"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/integration/util"
)

// 1.2 code gets:
// 	quota_test.go:95: Took 4.218619579s to scale up without quota
// 	quota_test.go:199: unexpected error: timed out waiting for the condition, ended with 342 pods (1 minute)
// 1.3+ code gets:
// 	quota_test.go:100: Took 4.196205966s to scale up without quota
// 	quota_test.go:115: Took 12.021640372s to scale up with quota
func TestQuota(t *testing.T) {
	// Set up a master
	h := &framework.MasterHolder{Initialized: make(chan struct{})}
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		<-h.Initialized
		h.M.GenericAPIServer.Handler.ServeHTTP(w, req)
	}))

	admissionCh := make(chan struct{})
	clientConfig := &restclient.Config{QPS: -1, Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}}
	clientset := clientset.NewForConfigOrDie(clientConfig)
	config := &resourcequotaapi.Configuration{}
	admission, err := resourcequota.NewResourceQuota(config, 5, admissionCh)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	admission.SetExternalKubeClientSet(clientset)
	internalInformers := informers.NewSharedInformerFactory(clientset, controller.NoResyncPeriodFunc())
	admission.SetExternalKubeInformerFactory(internalInformers)
	qca := quotainstall.NewQuotaConfigurationForAdmission()
	admission.SetQuotaConfiguration(qca)
	defer close(admissionCh)

	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.GenericConfig.AdmissionControl = admission
	_, _, closeFn := framework.RunAnApiserverUsingServer(masterConfig, s, h)
	defer closeFn()

	ns := framework.CreateTestingNamespace("quotaed", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	ns2 := framework.CreateTestingNamespace("non-quotaed", s, t)
	defer framework.DeleteTestingNamespace(ns2, s, t)

	controllerCh := make(chan struct{})
	defer close(controllerCh)

	metadataClient := metadata.NewForConfigOrDie(clientConfig)
	metadataInformers := metadatainformer.NewSharedInformerFactoryWithOptions(metadataClient, controller.NoResyncPeriodFunc(), metadatainformer.WithStopOnZeroEventHandlers(true))
	informers := informerfactory.NewInformerFactory(internalInformers, metadataInformers)
	rm := replicationcontroller.NewReplicationManager(
		internalInformers.Core().V1().Pods(),
		internalInformers.Core().V1().ReplicationControllers(),
		clientset,
		replicationcontroller.BurstReplicas,
	)
	rm.SetEventRecorder(&record.FakeRecorder{})
	go rm.Run(3, controllerCh)

	discoveryFunc := clientset.Discovery().ServerPreferredNamespacedResources
	listerFuncForResource := generic.ListerFuncForResourceFunc(informers.ForResource)
	qc := quotainstall.NewQuotaConfigurationForControllers(listerFuncForResource)
	informersStarted := make(chan struct{})
	resourceQuotaControllerOptions := &resourcequotacontroller.ControllerOptions{
		QuotaClient:               clientset.CoreV1(),
		ResourceQuotaInformer:     internalInformers.Core().V1().ResourceQuotas(),
		ResyncPeriod:              controller.NoResyncPeriodFunc,
		InformerFactory:           informers,
		ReplenishmentResyncPeriod: controller.NoResyncPeriodFunc,
		DiscoveryFunc:             discoveryFunc,
		IgnoredResourcesFunc:      qc.IgnoredResources,
		InformersStarted:          informersStarted,
		Registry:                  generic.NewRegistry(qc.Evaluators()),
	}
	resourceQuotaController, err := resourcequotacontroller.NewController(resourceQuotaControllerOptions)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	go resourceQuotaController.Run(2, controllerCh)

	// Periodically the quota controller to detect new resource types
	go resourceQuotaController.Sync(discoveryFunc, 30*time.Second, controllerCh)

	internalInformers.Start(controllerCh)
	informers.Start(controllerCh)
	close(informersStarted)

	startTime := time.Now()
	scale(t, ns2.Name, clientset)
	endTime := time.Now()
	t.Logf("Took %v to scale up without quota", endTime.Sub(startTime))

	quota := &v1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "quota",
			Namespace: ns.Name,
		},
		Spec: v1.ResourceQuotaSpec{
			Hard: v1.ResourceList{
				v1.ResourcePods: resource.MustParse("1000"),
			},
		},
	}
	waitForQuota(t, quota, clientset)

	startTime = time.Now()
	scale(t, "quotaed", clientset)
	endTime = time.Now()
	t.Logf("Took %v to scale up with quota", endTime.Sub(startTime))
}

func waitForQuota(t *testing.T, quota *v1.ResourceQuota, clientset clientset.Interface) {
	w, err := clientset.CoreV1().ResourceQuotas(quota.Namespace).Watch(context.TODO(), metav1.SingleObject(metav1.ObjectMeta{Name: quota.Name}))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	q, err := clientset.CoreV1().ResourceQuotas(quota.Namespace).Create(context.TODO(), quota, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	t.Logf("created quota %v", q)

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
	defer cancel()
	_, err = watchtools.UntilWithoutRetry(ctx, w, func(event watch.Event) (bool, error) {
		switch event.Type {
		case watch.Modified:
		default:
			return false, nil
		}
		switch cast := event.Object.(type) {
		case *v1.ResourceQuota:
			if len(cast.Status.Hard) > 0 {
				return true, nil
			}
		}

		return false, nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func scale(t *testing.T, namespace string, clientset *clientset.Clientset) {
	target := int32(100)
	rc := &v1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: namespace,
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: &target,
			Selector: map[string]string{"foo": "bar"},
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"foo": "bar",
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "container",
							Image: "busybox",
						},
					},
				},
			},
		},
	}

	w, err := clientset.CoreV1().ReplicationControllers(namespace).Watch(context.TODO(), metav1.SingleObject(metav1.ObjectMeta{Name: rc.Name}))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if _, err := clientset.CoreV1().ReplicationControllers(namespace).Create(context.TODO(), rc, metav1.CreateOptions{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	_, err = watchtools.UntilWithoutRetry(ctx, w, func(event watch.Event) (bool, error) {
		switch event.Type {
		case watch.Modified:
		default:
			return false, nil
		}

		switch cast := event.Object.(type) {
		case *v1.ReplicationController:
			fmt.Printf("Found %v of %v replicas\n", int(cast.Status.Replicas), target)
			if cast.Status.Replicas == target {
				return true, nil
			}
		}

		return false, nil
	})
	if err != nil {
		pods, _ := clientset.CoreV1().Pods(namespace).List(context.TODO(), metav1.ListOptions{LabelSelector: labels.Everything().String(), FieldSelector: fields.Everything().String()})
		t.Fatalf("unexpected error: %v, ended with %v pods", err, len(pods.Items))
	}
}

func TestQuotaLimitedResourceDenial(t *testing.T) {
	// Set up a master
	h := &framework.MasterHolder{Initialized: make(chan struct{})}
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		<-h.Initialized
		h.M.GenericAPIServer.Handler.ServeHTTP(w, req)
	}))

	admissionCh := make(chan struct{})
	clientset := clientset.NewForConfigOrDie(&restclient.Config{QPS: -1, Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})

	// stop creation of a pod resource unless there is a quota
	config := &resourcequotaapi.Configuration{
		LimitedResources: []resourcequotaapi.LimitedResource{
			{
				Resource:      "pods",
				MatchContains: []string{"pods"},
			},
		},
	}
	qca := quotainstall.NewQuotaConfigurationForAdmission()
	admission, err := resourcequota.NewResourceQuota(config, 5, admissionCh)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	admission.SetExternalKubeClientSet(clientset)
	externalInformers := informers.NewSharedInformerFactory(clientset, controller.NoResyncPeriodFunc())
	admission.SetExternalKubeInformerFactory(externalInformers)
	admission.SetQuotaConfiguration(qca)
	defer close(admissionCh)

	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.GenericConfig.AdmissionControl = admission
	_, _, closeFn := framework.RunAnApiserverUsingServer(masterConfig, s, h)
	defer closeFn()

	ns := framework.CreateTestingNamespace("quota", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	controllerCh := make(chan struct{})
	defer close(controllerCh)

	informers := informers.NewSharedInformerFactory(clientset, controller.NoResyncPeriodFunc())
	rm := replicationcontroller.NewReplicationManager(
		informers.Core().V1().Pods(),
		informers.Core().V1().ReplicationControllers(),
		clientset,
		replicationcontroller.BurstReplicas,
	)
	rm.SetEventRecorder(&record.FakeRecorder{})
	go rm.Run(3, controllerCh)

	discoveryFunc := clientset.Discovery().ServerPreferredNamespacedResources
	listerFuncForResource := generic.ListerFuncForResourceFunc(informers.ForResource)
	qc := quotainstall.NewQuotaConfigurationForControllers(listerFuncForResource)
	informersStarted := make(chan struct{})
	resourceQuotaControllerOptions := &resourcequotacontroller.ControllerOptions{
		QuotaClient:               clientset.CoreV1(),
		ResourceQuotaInformer:     informers.Core().V1().ResourceQuotas(),
		ResyncPeriod:              controller.NoResyncPeriodFunc,
		InformerFactory:           informers,
		ReplenishmentResyncPeriod: controller.NoResyncPeriodFunc,
		DiscoveryFunc:             discoveryFunc,
		IgnoredResourcesFunc:      qc.IgnoredResources,
		InformersStarted:          informersStarted,
		Registry:                  generic.NewRegistry(qc.Evaluators()),
	}
	resourceQuotaController, err := resourcequotacontroller.NewController(resourceQuotaControllerOptions)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	go resourceQuotaController.Run(2, controllerCh)

	// Periodically the quota controller to detect new resource types
	go resourceQuotaController.Sync(discoveryFunc, 30*time.Second, controllerCh)

	externalInformers.Start(controllerCh)
	informers.Start(controllerCh)
	close(informersStarted)

	// try to create a pod
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: ns.Name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "container",
					Image: "busybox",
				},
			},
		},
	}
	if _, err := clientset.CoreV1().Pods(ns.Name).Create(context.TODO(), pod, metav1.CreateOptions{}); err == nil {
		t.Fatalf("expected error for insufficient quota")
	}

	// now create a covering quota
	// note: limited resource does a matchContains, so we now have "pods" matching "pods" and "count/pods"
	quota := &v1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "quota",
			Namespace: ns.Name,
		},
		Spec: v1.ResourceQuotaSpec{
			Hard: v1.ResourceList{
				v1.ResourcePods:               resource.MustParse("1000"),
				v1.ResourceName("count/pods"): resource.MustParse("1000"),
			},
		},
	}
	waitForQuota(t, quota, clientset)

	// attempt to create a new pod once the quota is propagated
	err = wait.PollImmediate(5*time.Second, time.Minute, func() (bool, error) {
		// retry until we succeed (to allow time for all changes to propagate)
		if _, err := clientset.CoreV1().Pods(ns.Name).Create(context.TODO(), pod, metav1.CreateOptions{}); err == nil {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// TestCRDQuotaUninstalledResources confirms that the quota controller continues to operate on CRDs that
// are uninstalled and reinstalled on the cluster between quota controller syncs.
func TestCRDQuotaUninstallReinstall(t *testing.T) {
	ctx := setup(t, 2)
	defer ctx.tearDown()
	// install crd on cluster
	ns := util.CreateNamespaceOrDie("test-crd", ctx.clientSet, t)
	definition, resourceClient := util.CreateRandomCustomResourceDefinition(t, ctx.apiExtensionClient, ctx.dynamicClient, ns.Name)
	time.Sleep(ctx.syncPeriod)
	n := 3
	quotaName := "quota"
	hard := v1.ResourceList{}
	resourceName := v1.ResourceName(fmt.Sprintf("count/%s.%s", definition.Spec.Names.Plural, definition.Spec.Group))
	hard[resourceName] = resource.MustParse(strconv.Itoa(n))
	quota := &v1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{
			Name:      quotaName,
			Namespace: ns.Name,
		},
		Spec: v1.ResourceQuotaSpec{
			Hard: hard,
		},
	}
	waitForQuota(t, quota, ctx.clientSet)
	// create the crd instance
	_, err := resourceClient.Create(context.TODO(), util.NewCRDInstance(definition, ns.Name, fmt.Sprintf("crd-%d", 1)), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create crd instance 1, err: %v", err)
	}

	// confirm crd instance shows up in quota used count
	err = pollExpectedQuota(ctx.clientSet, 1, resourceName, quotaName, ns.Name)
	if err != nil {
		t.Fatalf("err %v", err)
	}

	// retrieve the informer before deletion to check it's been stopped.
	gvr := schema.GroupVersionResource{Group: definition.Spec.Group, Version: definition.Spec.Versions[0].Name, Resource: definition.Spec.Names.Plural}
	informer, err := ctx.informers.ForResource(gvr)
	if err != nil {
		t.Fatalf("error getting informer for resource")
	}
	if informer.Informer().IsStopped() {
		t.Fatalf("informer stopped unexpectedly prior to deletion")
	}

	// delete the definition which will cascade to the instance
	if err := apiextensionstestserver.DeleteV1CustomResourceDefinition(definition, ctx.apiExtensionClient); err != nil {
		t.Fatalf("failed to delete %q: %v", definition.Name, err)
	}

	// ensure the informer is stopped upon CRD deletion
	if err := wait.Poll(1*time.Second, 30*time.Second, func() (bool, error) {
		return informer.Informer().IsStopped(), nil
	}); err != nil {
		t.Fatalf("CRD informer failed to stop upon deletion")
	}

	// check quota used count is now 0 after deletion
	err = pollExpectedQuota(ctx.clientSet, 0, resourceName, quotaName, ns.Name)
	if err != nil {
		t.Fatalf("failed to reset quota count after CRD deletion")
	}

	// reinstall the definition
	accessor := meta.NewAccessor()
	accessor.SetResourceVersion(definition, "")
	_, err = apiextensionstestserver.CreateNewV1CustomResourceDefinition(definition, ctx.apiExtensionClient, ctx.dynamicClient)
	if err != nil {
		t.Fatalf("failed to create CustomResourceDefinition on round 2: %v", err)
	}

	// create instance
	_, err = resourceClient.Create(context.TODO(), util.NewCRDInstance(definition, ns.Name, fmt.Sprintf("crd-%d", 2)), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create crd instance 2, err: %v", err)
	}

	// confirm quota is updated with new instance in used count
	err = pollExpectedQuota(ctx.clientSet, 1, resourceName, quotaName, ns.Name)
	if err != nil {
		t.Fatalf("err %v", err)
	}
}

func TestCRDQuota(t *testing.T) {
	ctx := setup(t, 2)
	defer ctx.tearDown()

	// install crd on cluster, give it time to sync
	ns := util.CreateNamespaceOrDie("test-crd", ctx.clientSet, t)
	definition, resourceClient := util.CreateRandomCustomResourceDefinition(t, ctx.apiExtensionClient, ctx.dynamicClient, ns.Name)
	time.Sleep(ctx.syncPeriod)

	// create crd quota N
	n := 3
	quotaName := "quota"
	hard := v1.ResourceList{}
	resourceName := v1.ResourceName(fmt.Sprintf("count/%s.%s", definition.Spec.Names.Plural, definition.Spec.Group))
	hard[resourceName] = resource.MustParse(strconv.Itoa(n))
	quota := &v1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{
			Name:      quotaName,
			Namespace: ns.Name,
		},
		Spec: v1.ResourceQuotaSpec{
			Hard: hard,
		},
	}
	waitForQuota(t, quota, ctx.clientSet)
	// create n+1 crd objects, confirm failure of the last one
	for i := 1; i < n+1; i++ {
		// create a new crd instance
		_, err := resourceClient.Create(context.TODO(), util.NewCRDInstance(definition, ns.Name, fmt.Sprintf("crd-%d", i)), metav1.CreateOptions{})
		if err != nil {
			if i != n+1 {
				t.Fatalf("failed to create crd instance on iteration: %d, err: %v", i, err)
			}
		} else {
			// confirm that eventually, the resource quota used field
			// equals the number of crd instances created
			err := pollExpectedQuota(ctx.clientSet, i, resourceName, quotaName, ns.Name)
			if err != nil {
				t.Fatalf("failed polling the expected quota:  %v", err)
			}
		}
	}

}

func pollExpectedQuota(clientSet kubernetes.Interface, target int, resourceName v1.ResourceName, quotaName, ns string) error {
	return wait.Poll(time.Second, time.Minute, func() (bool, error) {
		q, err := clientSet.CoreV1().ResourceQuotas(ns).Get(context.TODO(), quotaName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		used, ok := q.Status.Used[resourceName]
		if ok && used.AsDec().UnscaledBig().Int64() == int64(target) {
			return true, nil
		}
		return false, nil
	})

}

type testContext struct {
	tearDown           func()
	rq                 *resourcequotacontroller.Controller
	clientSet          clientset.Interface
	apiExtensionClient apiextensionsclientset.Interface
	dynamicClient      dynamic.Interface
	metadataClient     metadata.Interface
	startRQ            func(workers int)
	informers          informerfactory.InformerFactory
	// syncPeriod is how often the RQ started with startRQ will be resynced.
	syncPeriod time.Duration
}

// if workerCount > 0, will start the RQ, otherwise it's up to the caller to Run() the RQ.
func setup(t *testing.T, workerCount int) *testContext {
	return setupWithServer(t, kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd()), workerCount)
}

func setupWithServer(t *testing.T, result *kubeapiservertesting.TestServer, workerCount int) *testContext {
	ctx := util.ExtensionSetup(t, result, workerCount)
	rm := replicationcontroller.NewReplicationManager(
		ctx.SharedInformers.Core().V1().Pods(),
		ctx.SharedInformers.Core().V1().ReplicationControllers(),
		ctx.ClientSet,
		replicationcontroller.BurstReplicas,
	)
	rm.SetEventRecorder(&record.FakeRecorder{})
	go rm.Run(3, ctx.StopCh)

	discoveryFunc := ctx.ClientSet.Discovery().ServerPreferredNamespacedResources
	listerFuncForResource := generic.ListerFuncForResourceFunc(ctx.SharedInformers.ForResource)
	qc := quotainstall.NewQuotaConfigurationForControllers(listerFuncForResource)
	informersStarted := make(chan struct{})
	informers := informerfactory.NewInformerFactory(ctx.SharedInformers, ctx.MetadataInformers)
	resourceQuotaControllerOptions := &resourcequotacontroller.ControllerOptions{
		QuotaClient:               ctx.ClientSet.CoreV1(),
		ResourceQuotaInformer:     ctx.SharedInformers.Core().V1().ResourceQuotas(),
		ResyncPeriod:              controller.NoResyncPeriodFunc,
		InformerFactory:           informers,
		ReplenishmentResyncPeriod: controller.NoResyncPeriodFunc,
		DiscoveryFunc:             discoveryFunc,
		IgnoredResourcesFunc:      qc.IgnoredResources,
		InformersStarted:          informersStarted,
		Registry:                  generic.NewRegistry(qc.Evaluators()),
	}
	resourceQuotaController, err := resourcequotacontroller.NewController(resourceQuotaControllerOptions)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}

	syncPeriod := 5 * time.Second
	startRQ := func(workers int) {
		go wait.Until(func() {
			ctx.RESTMapper.Reset()
		}, syncPeriod, ctx.StopCh)
		go resourceQuotaController.Run(1, ctx.StopCh)
		go resourceQuotaController.Sync(discoveryFunc, syncPeriod, ctx.StopCh)
	}
	if workerCount > 0 {
		startRQ(workerCount)
		informers.Start(ctx.StopCh)
		close(informersStarted)
	}
	return &testContext{
		tearDown:           ctx.TearDown,
		rq:                 resourceQuotaController,
		clientSet:          ctx.ClientSet,
		apiExtensionClient: ctx.APIExtensionClient,
		dynamicClient:      ctx.DynamicClient,
		metadataClient:     ctx.MetadataClient,
		startRQ:            startRQ,
		informers:          informers,
		syncPeriod:         syncPeriod,
	}
}
