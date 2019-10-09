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

package volumescheduling

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/client-go/informers"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/events"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/scheduler"
	schedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/test/integration/framework"

	// Install "DefaultProvider" algorithprovider
	_ "k8s.io/kubernetes/pkg/scheduler/algorithmprovider/defaults"
)

type testContext struct {
	closeFn         framework.CloseFunc
	httpServer      *httptest.Server
	ns              *v1.Namespace
	clientSet       *clientset.Clientset
	informerFactory informers.SharedInformerFactory
	scheduler       *scheduler.Scheduler
	stopCh          chan struct{}
}

// initTestMaster initializes a test environment and creates a master with default
// configuration.
func initTestMaster(t *testing.T, nsPrefix string, admission admission.Interface) *testContext {
	context := testContext{
		stopCh: make(chan struct{}),
	}

	// 1. Create master
	h := &framework.MasterHolder{Initialized: make(chan struct{})}
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		<-h.Initialized
		h.M.GenericAPIServer.Handler.ServeHTTP(w, req)
	}))

	masterConfig := framework.NewIntegrationTestMasterConfig()

	if admission != nil {
		masterConfig.GenericConfig.AdmissionControl = admission
	}

	_, context.httpServer, context.closeFn = framework.RunAMasterUsingServer(masterConfig, s, h)

	if nsPrefix != "default" {
		context.ns = framework.CreateTestingNamespace(nsPrefix+string(uuid.NewUUID()), s, t)
	} else {
		context.ns = framework.CreateTestingNamespace("default", s, t)
	}

	// 2. Create kubeclient
	context.clientSet = clientset.NewForConfigOrDie(
		&restclient.Config{
			QPS: -1, Host: s.URL,
			ContentConfig: restclient.ContentConfig{
				GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"},
			},
		},
	)
	return &context
}

// initTestSchedulerWithOptions initializes a test environment and creates a scheduler with default
// configuration and other options.
func initTestSchedulerWithOptions(
	t *testing.T,
	context *testContext,
	resyncPeriod time.Duration,
) *testContext {
	// 1. Create scheduler
	context.informerFactory = informers.NewSharedInformerFactory(context.clientSet, resyncPeriod)

	podInformer := context.informerFactory.Core().V1().Pods()
	eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{
		Interface: context.clientSet.EventsV1beta1().Events(""),
	})
	recorder := eventBroadcaster.NewRecorder(
		legacyscheme.Scheme,
		v1.DefaultSchedulerName,
	)

	var err error
	context.scheduler, err = createSchedulerWithPodInformer(
		context.clientSet, podInformer, context.informerFactory, recorder, context.stopCh)

	if err != nil {
		t.Fatalf("Couldn't create scheduler: %v", err)
	}

	stopCh := make(chan struct{})
	eventBroadcaster.StartRecordingToSink(stopCh)

	context.informerFactory.Start(context.scheduler.StopEverything)
	context.informerFactory.WaitForCacheSync(context.scheduler.StopEverything)

	context.scheduler.Run()
	return context
}

// createSchedulerWithPodInformer creates a new scheduler.
func createSchedulerWithPodInformer(
	clientSet clientset.Interface,
	podInformer coreinformers.PodInformer,
	informerFactory informers.SharedInformerFactory,
	recorder events.EventRecorder,
	stopCh <-chan struct{},
) (*scheduler.Scheduler, error) {
	defaultProviderName := schedulerconfig.SchedulerDefaultProviderName

	return scheduler.New(
		clientSet,
		informerFactory,
		podInformer,
		recorder,
		schedulerconfig.SchedulerAlgorithmSource{
			Provider: &defaultProviderName,
		},
		stopCh,
	)
}

// cleanupTest deletes the scheduler and the test namespace. It should be called
// at the end of a test.
func cleanupTest(t *testing.T, context *testContext) {
	// Kill the scheduler.
	close(context.stopCh)
	// Cleanup nodes.
	context.clientSet.CoreV1().Nodes().DeleteCollection(nil, metav1.ListOptions{})
	framework.DeleteTestingNamespace(context.ns, context.httpServer, t)
	context.closeFn()
}

// waitForPodToScheduleWithTimeout waits for a pod to get scheduled and returns
// an error if it does not scheduled within the given timeout.
func waitForPodToScheduleWithTimeout(cs clientset.Interface, pod *v1.Pod, timeout time.Duration) error {
	return wait.Poll(100*time.Millisecond, timeout, podScheduled(cs, pod.Namespace, pod.Name))
}

// waitForPodToSchedule waits for a pod to get scheduled and returns an error if
// it does not get scheduled within the timeout duration (30 seconds).
func waitForPodToSchedule(cs clientset.Interface, pod *v1.Pod) error {
	return waitForPodToScheduleWithTimeout(cs, pod, 30*time.Second)
}

// waitForPodUnscheduleWithTimeout waits for a pod to fail scheduling and returns
// an error if it does not become unschedulable within the given timeout.
func waitForPodUnschedulableWithTimeout(cs clientset.Interface, pod *v1.Pod, timeout time.Duration) error {
	return wait.Poll(100*time.Millisecond, timeout, podUnschedulable(cs, pod.Namespace, pod.Name))
}

// waitForPodUnschedule waits for a pod to fail scheduling and returns
// an error if it does not become unschedulable within the timeout duration (30 seconds).
func waitForPodUnschedulable(cs clientset.Interface, pod *v1.Pod) error {
	return waitForPodUnschedulableWithTimeout(cs, pod, 30*time.Second)
}

// podScheduled returns true if a node is assigned to the given pod.
func podScheduled(c clientset.Interface, podNamespace, podName string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(podNamespace).Get(podName, metav1.GetOptions{})
		if err != nil {
			// This could be a connection error so we want to retry.
			return false, nil
		}
		if pod.Spec.NodeName == "" {
			return false, nil
		}
		return true, nil
	}
}

// podUnschedulable returns a condition function that returns true if the given pod
// gets unschedulable status.
func podUnschedulable(c clientset.Interface, podNamespace, podName string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(podNamespace).Get(podName, metav1.GetOptions{})
		if err != nil {
			// This could be a connection error so we want to retry.
			return false, nil
		}
		_, cond := podutil.GetPodCondition(&pod.Status, v1.PodScheduled)
		return cond != nil && cond.Status == v1.ConditionFalse &&
			cond.Reason == v1.PodReasonUnschedulable, nil
	}
}
