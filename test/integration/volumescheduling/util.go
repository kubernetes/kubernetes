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
	"context"
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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/events"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controlplane"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	"k8s.io/kubernetes/test/integration/framework"
)

type testContext struct {
	closeFn         framework.CloseFunc
	httpServer      *httptest.Server
	ns              *v1.Namespace
	clientSet       *clientset.Clientset
	informerFactory informers.SharedInformerFactory
	scheduler       *scheduler.Scheduler

	ctx      context.Context
	cancelFn context.CancelFunc
}

// initTestMaster initializes a test environment and creates a master with default
// configuration. Alpha resources are enabled automatically if the corresponding feature
// is enabled.
func initTestMaster(t *testing.T, nsPrefix string, admission admission.Interface) *testContext {
	ctx, cancelFunc := context.WithCancel(context.Background())
	testCtx := testContext{
		ctx:      ctx,
		cancelFn: cancelFunc,
	}

	// 1. Create master
	h := &framework.MasterHolder{Initialized: make(chan struct{})}
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		<-h.Initialized
		h.M.GenericAPIServer.Handler.ServeHTTP(w, req)
	}))

	masterConfig := framework.NewIntegrationTestMasterConfig()
	resourceConfig := controlplane.DefaultAPIResourceConfigSource()
	if utilfeature.DefaultFeatureGate.Enabled(features.CSIStorageCapacity) {
		resourceConfig.EnableVersions(schema.GroupVersion{
			Group:   "storage.k8s.io",
			Version: "v1alpha1",
		})
	}
	masterConfig.ExtraConfig.APIResourceConfigSource = resourceConfig

	if admission != nil {
		masterConfig.GenericConfig.AdmissionControl = admission
	}

	_, testCtx.httpServer, testCtx.closeFn = framework.RunAMasterUsingServer(masterConfig, s, h)

	if nsPrefix != "default" {
		testCtx.ns = framework.CreateTestingNamespace(nsPrefix+string(uuid.NewUUID()), s, t)
	} else {
		testCtx.ns = framework.CreateTestingNamespace("default", s, t)
	}

	// 2. Create kubeclient
	testCtx.clientSet = clientset.NewForConfigOrDie(
		&restclient.Config{
			QPS: -1, Host: s.URL,
			ContentConfig: restclient.ContentConfig{
				GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"},
			},
		},
	)
	return &testCtx
}

// initTestSchedulerWithOptions initializes a test environment and creates a scheduler with default
// configuration and other options.
func initTestSchedulerWithOptions(
	t *testing.T,
	testCtx *testContext,
	resyncPeriod time.Duration,
) *testContext {
	// 1. Create scheduler
	testCtx.informerFactory = informers.NewSharedInformerFactory(testCtx.clientSet, resyncPeriod)

	eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{
		Interface: testCtx.clientSet.EventsV1(),
	})

	var err error
	testCtx.scheduler, err = scheduler.New(
		testCtx.clientSet,
		testCtx.informerFactory,
		profile.NewRecorderFactory(eventBroadcaster),
		testCtx.ctx.Done())

	if err != nil {
		t.Fatalf("Couldn't create scheduler: %v", err)
	}

	eventBroadcaster.StartRecordingToSink(testCtx.ctx.Done())

	testCtx.informerFactory.Start(testCtx.scheduler.StopEverything)
	testCtx.informerFactory.WaitForCacheSync(testCtx.scheduler.StopEverything)

	go testCtx.scheduler.Run(testCtx.ctx)
	return testCtx
}

// cleanupTest deletes the scheduler and the test namespace. It should be called
// at the end of a test.
func cleanupTest(t *testing.T, testCtx *testContext) {
	// Kill the scheduler.
	testCtx.cancelFn()
	// Cleanup nodes.
	testCtx.clientSet.CoreV1().Nodes().DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{})
	framework.DeleteTestingNamespace(testCtx.ns, testCtx.httpServer, t)
	testCtx.closeFn()
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
		pod, err := c.CoreV1().Pods(podNamespace).Get(context.TODO(), podName, metav1.GetOptions{})
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
		pod, err := c.CoreV1().Pods(podNamespace).Get(context.TODO(), podName, metav1.GetOptions{})
		if err != nil {
			// This could be a connection error so we want to retry.
			return false, nil
		}
		_, cond := podutil.GetPodCondition(&pod.Status, v1.PodScheduled)
		return cond != nil && cond.Status == v1.ConditionFalse &&
			cond.Reason == v1.PodReasonUnschedulable, nil
	}
}
