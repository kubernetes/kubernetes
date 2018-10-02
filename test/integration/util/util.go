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

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	clientv1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	"k8s.io/kubernetes/pkg/scheduler/factory"
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
		glog.Infof("destroying API server")
		s.Close()
		glog.Infof("destroyed API server")
	}
	return s.URL, shutdownFunc
}

// StartScheduler configures and starts a scheduler given a handle to the clientSet interface
// and event broadcaster. It returns a handle to the configurator for the running scheduler
// and the shutdown function to stop it.
func StartScheduler(clientSet clientset.Interface) (scheduler.Configurator, ShutdownFunc) {
	informerFactory := informers.NewSharedInformerFactory(clientSet, 0)

	evtBroadcaster := record.NewBroadcaster()
	evtWatch := evtBroadcaster.StartRecordingToSink(&clientv1core.EventSinkImpl{
		Interface: clientSet.CoreV1().Events("")})

	schedulerConfigurator := createSchedulerConfigurator(clientSet, informerFactory)

	sched, err := scheduler.NewFromConfigurator(schedulerConfigurator, func(conf *scheduler.Config) {
		conf.Recorder = evtBroadcaster.NewRecorder(legacyscheme.Scheme, v1.EventSource{Component: "scheduler"})
	})
	if err != nil {
		glog.Fatalf("Error creating scheduler: %v", err)
	}

	stop := make(chan struct{})
	informerFactory.Start(stop)

	sched.Run()

	shutdownFunc := func() {
		glog.Infof("destroying scheduler")
		evtWatch.Stop()
		sched.StopEverything()
		close(stop)
		glog.Infof("destroyed scheduler")
	}
	return schedulerConfigurator, shutdownFunc
}

// createSchedulerConfigurator create a configurator for scheduler with given informer factory and default name.
func createSchedulerConfigurator(
	clientSet clientset.Interface,
	informerFactory informers.SharedInformerFactory,
) scheduler.Configurator {
	// Enable EnableEquivalenceClassCache for all integration tests.
	utilfeature.DefaultFeatureGate.Set("EnableEquivalenceClassCache=true")

	return factory.NewConfigFactory(&factory.ConfigFactoryArgs{
		SchedulerName:                  v1.DefaultSchedulerName,
		Client:                         clientSet,
		NodeInformer:                   informerFactory.Core().V1().Nodes(),
		PodInformer:                    informerFactory.Core().V1().Pods(),
		PvInformer:                     informerFactory.Core().V1().PersistentVolumes(),
		PvcInformer:                    informerFactory.Core().V1().PersistentVolumeClaims(),
		ReplicationControllerInformer:  informerFactory.Core().V1().ReplicationControllers(),
		ReplicaSetInformer:             informerFactory.Apps().V1().ReplicaSets(),
		StatefulSetInformer:            informerFactory.Apps().V1().StatefulSets(),
		ServiceInformer:                informerFactory.Core().V1().Services(),
		PdbInformer:                    informerFactory.Policy().V1beta1().PodDisruptionBudgets(),
		StorageClassInformer:           informerFactory.Storage().V1().StorageClasses(),
		HardPodAffinitySymmetricWeight: v1.DefaultHardPodAffinitySymmetricWeight,
		EnableEquivalenceClassCache:    utilfeature.DefaultFeatureGate.Enabled(features.EnableEquivalenceClassCache),
		DisablePreemption:              false,
		PercentageOfNodesToScore:       schedulerapi.DefaultPercentageOfNodesToScore,
	})
}
