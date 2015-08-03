/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package framework

import (
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/controller/replication"
	explatest "github.com/GoogleCloudPlatform/kubernetes/pkg/expapi/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/storage"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools/etcdtest"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/admission/admit"
	"github.com/golang/glog"
)

const (
	// Timeout used in benchmarks, to eg: scale an rc
	DefaultTimeout = 30 * time.Minute

	// Rc manifest used to create pods for benchmarks.
	// TODO: Convert this to a full path?
	TestRCManifest = "benchmark-controller.json"

	// Test Namspace, for pods and rcs.
	TestNS = "test"
)

// MasterComponents is a control struct for all master components started via NewMasterComponents.
// TODO: Include all master components (scheduler, nodecontroller).
// TODO: Reconcile with integration.go, currently the master used there doesn't understand
// how to restart cleanly, which is required for each iteration of a benchmark. The integration
// tests also don't make it easy to isolate and turn off components at will.
type MasterComponents struct {
	// Raw http server in front of the master
	ApiServer *httptest.Server
	// Kubernetes master, contains an embedded etcd storage
	KubeMaster *master.Master
	// Restclient used to talk to the kubernetes master
	RestClient *client.Client
	// Replication controller manager
	ControllerManager *replicationcontroller.ReplicationManager
	// Channel for stop signals to rc manager
	rcStopCh chan struct{}
	// Used to stop master components individually, and via MasterComponents.Stop
	once sync.Once
	// Kubernetes etcd storage, has embedded etcd client
	EtcdStorage storage.Interface
}

// Config is a struct of configuration directives for NewMasterComponents.
type Config struct {
	// If nil, a default is used, partially filled configs will not get populated.
	MasterConfig            *master.Config
	StartReplicationManager bool
	// If true, all existing etcd keys are purged before starting master components
	DeleteEtcdKeys bool
	// Client throttling qps
	QPS float32
	// Client burst qps, also burst replicas allowed in rc manager
	Burst int
	// TODO: Add configs for endpoints controller, scheduler etc
}

// NewMasterComponents creates, initializes and starts master components based on the given config.
func NewMasterComponents(c *Config) *MasterComponents {
	m, s, e := startMasterOrDie(c.MasterConfig)
	// TODO: Allow callers to pipe through a different master url and create a client/start components using it.
	glog.Infof("Master %+v", s.URL)
	if c.DeleteEtcdKeys {
		DeleteAllEtcdKeys()
	}
	restClient := client.NewOrDie(&client.Config{Host: s.URL, Version: testapi.Version(), QPS: c.QPS, Burst: c.Burst})
	rcStopCh := make(chan struct{})
	controllerManager := replicationcontroller.NewReplicationManager(restClient, c.Burst)

	// TODO: Support events once we can cleanly shutdown an event recorder.
	controllerManager.SetEventRecorder(&record.FakeRecorder{})
	if c.StartReplicationManager {
		go controllerManager.Run(runtime.NumCPU(), rcStopCh)
	}
	var once sync.Once
	return &MasterComponents{
		ApiServer:         s,
		KubeMaster:        m,
		RestClient:        restClient,
		ControllerManager: controllerManager,
		rcStopCh:          rcStopCh,
		EtcdStorage:       e,
		once:              once,
	}
}

// startMasterOrDie starts a kubernetes master and an httpserver to handle api requests
func startMasterOrDie(masterConfig *master.Config) (*master.Master, *httptest.Server, storage.Interface) {
	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))

	var etcdStorage storage.Interface
	var err error
	if masterConfig == nil {
		etcdClient := NewEtcdClient()
		etcdStorage, err = master.NewEtcdStorage(etcdClient, latest.InterfacesFor, latest.Version, etcdtest.PathPrefix())
		if err != nil {
			glog.Fatalf("Failed to create etcd storage for master %v", err)
		}
		expEtcdStorage, err := master.NewEtcdStorage(etcdClient, explatest.InterfacesFor, explatest.Version, etcdtest.PathPrefix())
		if err != nil {
			glog.Fatalf("Failed to create etcd storage for master %v", err)
		}

		masterConfig = &master.Config{
			DatabaseStorage:    etcdStorage,
			ExpDatabaseStorage: expEtcdStorage,
			KubeletClient:      client.FakeKubeletClient{},
			EnableLogsSupport:  false,
			EnableProfiling:    true,
			EnableUISupport:    false,
			APIPrefix:          "/api",
			ExpAPIPrefix:       "/experimental",
			Authorizer:         apiserver.NewAlwaysAllowAuthorizer(),
			AdmissionControl:   admit.NewAlwaysAdmit(),
		}
	} else {
		etcdStorage = masterConfig.DatabaseStorage
	}
	m = master.New(masterConfig)
	return m, s, etcdStorage
}

func (m *MasterComponents) stopRCManager() {
	close(m.rcStopCh)
}

func (m *MasterComponents) Stop(apiServer, rcManager bool) {
	glog.Infof("Stopping master components")
	if rcManager {
		// Ordering matters because the apiServer will only shutdown when pending
		// requests are done
		m.once.Do(m.stopRCManager)
	}
	if apiServer {
		m.ApiServer.Close()
	}
}

// RCFromManifest reads a .json file and returns the rc in it.
func RCFromManifest(fileName string) *api.ReplicationController {
	data, err := ioutil.ReadFile(fileName)
	if err != nil {
		glog.Fatalf("Unexpected error reading rc manifest %v", err)
	}
	var controller api.ReplicationController
	if err := api.Scheme.DecodeInto(data, &controller); err != nil {
		glog.Fatalf("Unexpected error reading rc manifest %v", err)
	}
	return &controller
}

// StopRC stops the rc via kubectl's stop library
func StopRC(rc *api.ReplicationController, restClient *client.Client) error {
	reaper, err := kubectl.ReaperFor("ReplicationController", restClient)
	if err != nil || reaper == nil {
		return err
	}
	_, err = reaper.Stop(rc.Namespace, rc.Name, 0, nil)
	if err != nil {
		return err
	}
	return nil
}

// ScaleRC scales the given rc to the given replicas.
func ScaleRC(name, ns string, replicas int, restClient *client.Client) (*api.ReplicationController, error) {
	scaler, err := kubectl.ScalerFor("ReplicationController", kubectl.NewScalerClient(restClient))
	if err != nil {
		return nil, err
	}
	retry := &kubectl.RetryParams{50 * time.Millisecond, DefaultTimeout}
	waitForReplicas := &kubectl.RetryParams{50 * time.Millisecond, DefaultTimeout}
	err = scaler.Scale(ns, name, uint(replicas), nil, retry, waitForReplicas)
	if err != nil {
		return nil, err
	}
	scaled, err := restClient.ReplicationControllers(ns).Get(name)
	if err != nil {
		return nil, err
	}
	return scaled, nil
}

// StartRC creates given rc if it doesn't already exist, then updates it via kubectl's scaler.
func StartRC(controller *api.ReplicationController, restClient *client.Client) (*api.ReplicationController, error) {
	created, err := restClient.ReplicationControllers(controller.Namespace).Get(controller.Name)
	if err != nil {
		glog.Infof("Rc %v doesn't exist, creating", controller.Name)
		created, err = restClient.ReplicationControllers(controller.Namespace).Create(controller)
		if err != nil {
			return nil, err
		}
	}
	// If we just created an rc, wait till it creates its replicas.
	return ScaleRC(created.Name, created.Namespace, controller.Spec.Replicas, restClient)
}

// StartPods check for numPods in TestNS. If they exist, it no-ops, otherwise it starts up
// a temp rc, scales it to match numPods, then deletes the rc leaving behind the pods.
func StartPods(numPods int, host string, restClient *client.Client) error {
	start := time.Now()
	defer func() {
		glog.Infof("StartPods took %v with numPods %d", time.Since(start), numPods)
	}()
	hostField := fields.OneTermEqualSelector(client.PodHost, host)
	pods, err := restClient.Pods(TestNS).List(labels.Everything(), hostField)
	if err != nil || len(pods.Items) == numPods {
		return err
	}
	glog.Infof("Found %d pods that match host %v, require %d", len(pods.Items), hostField, numPods)
	// For the sake of simplicity, assume all pods in TestNS have selectors matching TestRCManifest.
	controller := RCFromManifest(TestRCManifest)

	// Make the rc unique to the given host.
	controller.Spec.Replicas = numPods
	controller.Spec.Template.Spec.NodeName = host
	controller.Name = controller.Name + host
	controller.Spec.Selector["host"] = host
	controller.Spec.Template.Labels["host"] = host

	if rc, err := StartRC(controller, restClient); err != nil {
		return err
	} else {
		// Delete the rc, otherwise when we restart master components for the next benchmark
		// the rc controller will race with the pods controller in the rc manager.
		return restClient.ReplicationControllers(TestNS).Delete(rc.Name)
	}
}

// TODO: Merge this into startMasterOrDie.
func RunAMaster(t *testing.T) (*master.Master, *httptest.Server) {
	etcdClient := NewEtcdClient()
	etcdStorage, err := master.NewEtcdStorage(etcdClient, latest.InterfacesFor, testapi.Version(), etcdtest.PathPrefix())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expEtcdStorage, err := master.NewEtcdStorage(etcdClient, explatest.InterfacesFor, explatest.Version, etcdtest.PathPrefix())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	m := master.New(&master.Config{
		DatabaseStorage:    etcdStorage,
		ExpDatabaseStorage: expEtcdStorage,
		KubeletClient:      client.FakeKubeletClient{},
		EnableLogsSupport:  false,
		EnableProfiling:    true,
		EnableUISupport:    false,
		APIPrefix:          "/api",
		ExpAPIPrefix:       "/experimental",
		EnableExp:          true,
		Authorizer:         apiserver.NewAlwaysAllowAuthorizer(),
		AdmissionControl:   admit.NewAlwaysAdmit(),
	})

	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))

	return m, s
}

// Task is a function passed to worker goroutines by RunParallel.
// The function needs to implement its own thread safety.
type Task func(id int) error

// RunParallel spawns a goroutine per task in the given queue
func RunParallel(task Task, numTasks, numWorkers int) {
	start := time.Now()
	if numWorkers <= 0 {
		numWorkers = numTasks
	}
	defer func() {
		glog.Infof("RunParallel took %v for %d tasks and %d workers", time.Since(start), numTasks, numWorkers)
	}()
	var wg sync.WaitGroup
	semCh := make(chan struct{}, numWorkers)
	wg.Add(numTasks)
	for id := 0; id < numTasks; id++ {
		go func(id int) {
			semCh <- struct{}{}
			err := task(id)
			if err != nil {
				glog.Fatalf("Worker failed with %v", err)
			}
			<-semCh
			wg.Done()
		}(id)
	}
	wg.Wait()
	close(semCh)
}
