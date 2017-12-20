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

package benchmark

import (
	"net"
	"net/http"
	"net/http/httptest"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
	"k8s.io/api/core/v1"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	clientv1core "k8s.io/client-go/kubernetes/typed/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/plugin/pkg/scheduler"
	_ "k8s.io/kubernetes/plugin/pkg/scheduler/algorithmprovider"
	"k8s.io/kubernetes/plugin/pkg/scheduler/factory"
	"k8s.io/kubernetes/test/integration/framework"
)

const enableEquivalenceCache = true

// mustSetupScheduler starts the following components:
// - k8s api server (a.k.a. master)
// - scheduler
// It returns scheduler config factory and destroyFunc which should be used to
// remove resources after finished.
// Notes on rate limiter:
//   - client rate limit is set to 5000.
func mustSetupScheduler(metricServerEnabled bool) (schedulerConfigurator scheduler.Configurator, destroyFunc func()) {

	h := &framework.MasterHolder{Initialized: make(chan struct{})}
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		<-h.Initialized
		h.M.GenericAPIServer.Handler.ServeHTTP(w, req)
	}))

	if metricServerEnabled {
		go setupMetricsServer()
	}
	framework.RunAMasterUsingServer(framework.NewIntegrationTestMasterConfig(), s, h)

	clientSet := clientset.NewForConfigOrDie(&restclient.Config{
		Host:          s.URL,
		ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Groups[v1.GroupName].GroupVersion()},
		QPS:           5000.0,
		Burst:         5000,
	})

	informerFactory := informers.NewSharedInformerFactory(clientSet, 0)

	schedulerConfigurator = factory.NewConfigFactory(
		v1.DefaultSchedulerName,
		clientSet,
		informerFactory.Core().V1().Nodes(),
		informerFactory.Core().V1().Pods(),
		informerFactory.Core().V1().PersistentVolumes(),
		informerFactory.Core().V1().PersistentVolumeClaims(),
		informerFactory.Core().V1().ReplicationControllers(),
		informerFactory.Extensions().V1beta1().ReplicaSets(),
		informerFactory.Apps().V1beta1().StatefulSets(),
		informerFactory.Core().V1().Services(),
		informerFactory.Policy().V1beta1().PodDisruptionBudgets(),
		informerFactory.Storage().V1().StorageClasses(),
		v1.DefaultHardPodAffinitySymmetricWeight,
		enableEquivalenceCache,
	)

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartRecordingToSink(&clientv1core.EventSinkImpl{Interface: clientv1core.New(clientSet.CoreV1().RESTClient()).Events("")})

	sched, err := scheduler.NewFromConfigurator(schedulerConfigurator, func(conf *scheduler.Config) {
		conf.Recorder = eventBroadcaster.NewRecorder(legacyscheme.Scheme, v1.EventSource{Component: "scheduler"})
	})
	if err != nil {
		glog.Fatalf("Error creating scheduler: %v", err)
	}

	stop := make(chan struct{})
	informerFactory.Start(stop)

	sched.Run()

	destroyFunc = func() {
		glog.Infof("destroying")
		sched.StopEverything()
		close(stop)
		s.Close()
		glog.Infof("destroyed")
	}
	return
}

// setupMetricsServer starts the metrics server on port 9091.
func setupMetricsServer() {
	glog.Infof("starting metrics server")
	mux := http.NewServeMux()
	mux.Handle("/metrics", prometheus.Handler())
	server := &http.Server{
		// TODO: make the port configurable once we have yaml file.
		Addr:    net.JoinHostPort("127.0.0.1", "9091"),
		Handler: mux,
	}
	if err := server.ListenAndServe(); err != nil {
		glog.Errorf("Error creating metrics server. Either port: %v", err)
	}
	return
}
