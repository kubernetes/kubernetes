/*
Copyright 2018 The Kubernetes Authors.

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

package util

import (
	"net/http"
	"net/http/httptest"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/events"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/scheduler"

	// import DefaultProvider
	_ "k8s.io/kubernetes/pkg/scheduler/algorithmprovider/defaults"
	schedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/test/integration/framework"
)

// ShutdownFunc represents the function handle to be called, typically in a defer handler, to shutdown a running module
type ShutdownFunc func()

// StartApiserver starts a local API server for testing and returns the handle to the URL and the shutdown function to stop it.
func StartApiserver() (string, ShutdownFunc) {
	h := &framework.MasterHolder{Initialized: make(chan struct{})}
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		<-h.Initialized
		h.M.GenericAPIServer.Handler.ServeHTTP(w, req)
	}))

	framework.RunAMasterUsingServer(framework.NewIntegrationTestMasterConfig(), s, h)

	shutdownFunc := func() {
		klog.Infof("destroying API server")
		s.Close()
		klog.Infof("destroyed API server")
	}
	return s.URL, shutdownFunc
}

// StartScheduler configures and starts a scheduler given a handle to the clientSet interface
// and event broadcaster. It returns the running scheduler and the shutdown function to stop it.
func StartScheduler(clientSet clientset.Interface) (*scheduler.Scheduler, ShutdownFunc) {
	informerFactory := informers.NewSharedInformerFactory(clientSet, 0)
	stopCh := make(chan struct{})
	evtBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{
		Interface: clientSet.EventsV1beta1().Events("")})

	evtBroadcaster.StartRecordingToSink(stopCh)

	recorder := evtBroadcaster.NewRecorder(
		legacyscheme.Scheme,
		v1.DefaultSchedulerName,
	)

	sched, err := createScheduler(clientSet, informerFactory, recorder, stopCh)
	if err != nil {
		klog.Fatalf("Error creating scheduler: %v", err)
	}
	scheduler.AddAllEventHandlers(sched,
		v1.DefaultSchedulerName,
		informerFactory.Core().V1().Nodes(),
		informerFactory.Core().V1().Pods(),
		informerFactory.Core().V1().PersistentVolumes(),
		informerFactory.Core().V1().PersistentVolumeClaims(),
		informerFactory.Core().V1().Services(),
		informerFactory.Storage().V1().StorageClasses(),
		informerFactory.Storage().V1beta1().CSINodes(),
	)

	informerFactory.Start(stopCh)
	sched.Run()

	shutdownFunc := func() {
		klog.Infof("destroying scheduler")
		close(stopCh)
		klog.Infof("destroyed scheduler")
	}
	return sched, shutdownFunc
}

// createScheduler create a scheduler with given informer factory and default name.
func createScheduler(
	clientSet clientset.Interface,
	informerFactory informers.SharedInformerFactory,
	recorder events.EventRecorder,
	stopCh <-chan struct{},
) (*scheduler.Scheduler, error) {
	defaultProviderName := schedulerconfig.SchedulerDefaultProviderName

	return scheduler.New(
		clientSet,
		informerFactory.Core().V1().Nodes(),
		informerFactory.Core().V1().Pods(),
		informerFactory.Core().V1().PersistentVolumes(),
		informerFactory.Core().V1().PersistentVolumeClaims(),
		informerFactory.Core().V1().ReplicationControllers(),
		informerFactory.Apps().V1().ReplicaSets(),
		informerFactory.Apps().V1().StatefulSets(),
		informerFactory.Core().V1().Services(),
		informerFactory.Policy().V1beta1().PodDisruptionBudgets(),
		informerFactory.Storage().V1().StorageClasses(),
		informerFactory.Storage().V1beta1().CSINodes(),
		recorder,
		schedulerconfig.SchedulerAlgorithmSource{
			Provider: &defaultProviderName,
		},
		stopCh,
	)
}
