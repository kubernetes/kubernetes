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
	"net"
	"net/http"
	"net/http/httptest"
	goruntime "runtime"
	"sync"
	"testing"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apiserver"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller"
	replicationcontroller "k8s.io/kubernetes/pkg/controller/replication"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/genericapiserver"
	"k8s.io/kubernetes/pkg/kubectl"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/pkg/runtime"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/storage/etcd/etcdtest"
	"k8s.io/kubernetes/plugin/pkg/admission/admit"
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
	m, s := startMasterOrDie(c.MasterConfig)
	// TODO: Allow callers to pipe through a different master url and create a client/start components using it.
	glog.Infof("Master %+v", s.URL)
	if c.DeleteEtcdKeys {
		DeleteAllEtcdKeys()
	}
	// TODO: caesarxuchao: remove this client when the refactoring of client libraray is done.
	restClient := client.NewOrDie(&client.Config{Host: s.URL, ContentConfig: client.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}, QPS: c.QPS, Burst: c.Burst})
	clientset := clientset.NewForConfigOrDie(&client.Config{Host: s.URL, ContentConfig: client.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}, QPS: c.QPS, Burst: c.Burst})
	rcStopCh := make(chan struct{})
	controllerManager := replicationcontroller.NewReplicationManager(clientset, controller.NoResyncPeriodFunc, c.Burst, 4096)

	// TODO: Support events once we can cleanly shutdown an event recorder.
	controllerManager.SetEventRecorder(&record.FakeRecorder{})
	if c.StartReplicationManager {
		go controllerManager.Run(goruntime.NumCPU(), rcStopCh)
	}
	var once sync.Once
	return &MasterComponents{
		ApiServer:         s,
		KubeMaster:        m,
		RestClient:        restClient,
		ControllerManager: controllerManager,
		rcStopCh:          rcStopCh,
		once:              once,
	}
}

// startMasterOrDie starts a kubernetes master and an httpserver to handle api requests
func startMasterOrDie(masterConfig *master.Config) (*master.Master, *httptest.Server) {
	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))

	if masterConfig == nil {
		masterConfig = NewMasterConfig()
		masterConfig.EnableProfiling = true
		masterConfig.EnableSwaggerSupport = true
	}
	m, err := master.New(masterConfig)
	if err != nil {
		glog.Fatalf("error in bringing up the master: %v", err)
	}

	return m, s
}

// Returns a basic master config.
func NewMasterConfig() *master.Config {
	etcdClient := NewEtcdClient()
	storageVersions := make(map[string]string)

	etcdStorage := etcdstorage.NewEtcdStorage(etcdClient, testapi.Default.Codec(), etcdtest.PathPrefix(), false)
	storageVersions[api.GroupName] = testapi.Default.GroupVersion().String()
	autoscalingEtcdStorage := NewAutoscalingEtcdStorage(etcdClient)
	storageVersions[autoscaling.GroupName] = testapi.Autoscaling.GroupVersion().String()
	batchEtcdStorage := NewBatchEtcdStorage(etcdClient)
	storageVersions[batch.GroupName] = testapi.Batch.GroupVersion().String()
	expEtcdStorage := NewExtensionsEtcdStorage(etcdClient)
	storageVersions[extensions.GroupName] = testapi.Extensions.GroupVersion().String()

	storageDestinations := genericapiserver.NewStorageDestinations()
	storageDestinations.AddAPIGroup(api.GroupName, etcdStorage)
	storageDestinations.AddAPIGroup(autoscaling.GroupName, autoscalingEtcdStorage)
	storageDestinations.AddAPIGroup(batch.GroupName, batchEtcdStorage)
	storageDestinations.AddAPIGroup(extensions.GroupName, expEtcdStorage)

	return &master.Config{
		Config: &genericapiserver.Config{
			StorageDestinations: storageDestinations,
			StorageVersions:     storageVersions,
			APIPrefix:           "/api",
			APIGroupPrefix:      "/apis",
			Authorizer:          apiserver.NewAlwaysAllowAuthorizer(),
			AdmissionControl:    admit.NewAlwaysAdmit(),
			Serializer:          api.Codecs,
		},
		KubeletClient: kubeletclient.FakeKubeletClient{},
	}
}

// Returns the master config appropriate for most integration tests.
func NewIntegrationTestMasterConfig() *master.Config {
	masterConfig := NewMasterConfig()
	masterConfig.EnableCoreControllers = true
	masterConfig.EnableIndex = true
	masterConfig.PublicAddress = net.ParseIP("192.168.10.4")
	return masterConfig
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
		// TODO: Uncomment when fix #19254
		// m.ApiServer.Close()
	}
}

// RCFromManifest reads a .json file and returns the rc in it.
func RCFromManifest(fileName string) *api.ReplicationController {
	data, err := ioutil.ReadFile(fileName)
	if err != nil {
		glog.Fatalf("Unexpected error reading rc manifest %v", err)
	}
	var controller api.ReplicationController
	if err := runtime.DecodeInto(testapi.Default.Codec(), data, &controller); err != nil {
		glog.Fatalf("Unexpected error reading rc manifest %v", err)
	}
	return &controller
}

// StopRC stops the rc via kubectl's stop library
func StopRC(rc *api.ReplicationController, restClient *client.Client) error {
	reaper, err := kubectl.ReaperFor(api.Kind("ReplicationController"), restClient)
	if err != nil || reaper == nil {
		return err
	}
	err = reaper.Stop(rc.Namespace, rc.Name, 0, nil)
	if err != nil {
		return err
	}
	return nil
}

// ScaleRC scales the given rc to the given replicas.
func ScaleRC(name, ns string, replicas int, restClient *client.Client) (*api.ReplicationController, error) {
	scaler, err := kubectl.ScalerFor(api.Kind("ReplicationController"), restClient)
	if err != nil {
		return nil, err
	}
	retry := &kubectl.RetryParams{Interval: 50 * time.Millisecond, Timeout: DefaultTimeout}
	waitForReplicas := &kubectl.RetryParams{Interval: 50 * time.Millisecond, Timeout: DefaultTimeout}
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
	options := api.ListOptions{FieldSelector: hostField}
	pods, err := restClient.Pods(TestNS).List(options)
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
	masterConfig := NewMasterConfig()
	masterConfig.EnableProfiling = true
	m, err := master.New(masterConfig)
	if err != nil {
		// TODO: Return error.
		glog.Fatalf("error in bringing up the master: %v", err)
	}

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
