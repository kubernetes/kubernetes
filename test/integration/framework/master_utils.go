/*
Copyright 2014 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/apiserver"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/restclient"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller"
	replicationcontroller "k8s.io/kubernetes/pkg/controller/replication"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/genericapiserver"
	"k8s.io/kubernetes/pkg/kubectl"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage/storagebackend"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	"k8s.io/kubernetes/plugin/pkg/admission/admit"

	"github.com/pborman/uuid"
)

const (
	// Timeout used in benchmarks, to eg: scale an rc
	DefaultTimeout = 30 * time.Minute

	// Rc manifest used to create pods for benchmarks.
	// TODO: Convert this to a full path?
	TestRCManifest = "benchmark-controller.json"
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
	// TODO: caesarxuchao: remove this client when the refactoring of client libraray is done.
	restClient := client.NewOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}, QPS: c.QPS, Burst: c.Burst})
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}, QPS: c.QPS, Burst: c.Burst})
	rcStopCh := make(chan struct{})
	controllerManager := replicationcontroller.NewReplicationManagerFromClient(clientset, controller.NoResyncPeriodFunc, c.Burst, 4096)

	// TODO: Support events once we can cleanly shutdown an event recorder.
	controllerManager.SetEventRecorder(&record.FakeRecorder{})
	if c.StartReplicationManager {
		go controllerManager.Run(goruntime.NumCPU(), rcStopCh)
	}
	return &MasterComponents{
		ApiServer:         s,
		KubeMaster:        m,
		RestClient:        restClient,
		ControllerManager: controllerManager,
		rcStopCh:          rcStopCh,
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

func parseCIDROrDie(cidr string) *net.IPNet {
	_, parsed, err := net.ParseCIDR(cidr)
	if err != nil {
		glog.Fatalf("error while parsing CIDR: %s", cidr)
	}
	return parsed
}

// Returns a basic master config.
func NewMasterConfig() *master.Config {
	config := storagebackend.Config{
		ServerList: []string{GetEtcdURLFromEnv()},
		// This causes the integration tests to exercise the etcd
		// prefix code, so please don't change without ensuring
		// sufficient coverage in other ways.
		Prefix: uuid.New(),
	}

	negotiatedSerializer := NewSingleContentTypeSerializer(api.Scheme, testapi.Default.Codec(), runtime.ContentTypeJSON)

	storageFactory := genericapiserver.NewDefaultStorageFactory(config, runtime.ContentTypeJSON, negotiatedSerializer, genericapiserver.NewDefaultResourceEncodingConfig(), master.DefaultAPIResourceConfigSource())
	storageFactory.SetSerializer(
		unversioned.GroupResource{Group: api.GroupName, Resource: genericapiserver.AllResources},
		"",
		NewSingleContentTypeSerializer(api.Scheme, testapi.Default.Codec(), runtime.ContentTypeJSON))
	storageFactory.SetSerializer(
		unversioned.GroupResource{Group: autoscaling.GroupName, Resource: genericapiserver.AllResources},
		"",
		NewSingleContentTypeSerializer(api.Scheme, testapi.Autoscaling.Codec(), runtime.ContentTypeJSON))
	storageFactory.SetSerializer(
		unversioned.GroupResource{Group: batch.GroupName, Resource: genericapiserver.AllResources},
		"",
		NewSingleContentTypeSerializer(api.Scheme, testapi.Batch.Codec(), runtime.ContentTypeJSON))
	storageFactory.SetSerializer(
		unversioned.GroupResource{Group: apps.GroupName, Resource: genericapiserver.AllResources},
		"",
		NewSingleContentTypeSerializer(api.Scheme, testapi.Apps.Codec(), runtime.ContentTypeJSON))
	storageFactory.SetSerializer(
		unversioned.GroupResource{Group: extensions.GroupName, Resource: genericapiserver.AllResources},
		"",
		NewSingleContentTypeSerializer(api.Scheme, testapi.Extensions.Codec(), runtime.ContentTypeJSON))
	storageFactory.SetSerializer(
		unversioned.GroupResource{Group: policy.GroupName, Resource: genericapiserver.AllResources},
		"",
		NewSingleContentTypeSerializer(api.Scheme, testapi.Policy.Codec(), runtime.ContentTypeJSON))
	storageFactory.SetSerializer(
		unversioned.GroupResource{Group: rbac.GroupName, Resource: genericapiserver.AllResources},
		"",
		NewSingleContentTypeSerializer(api.Scheme, testapi.Rbac.Codec(), runtime.ContentTypeJSON))
	storageFactory.SetSerializer(
		unversioned.GroupResource{Group: certificates.GroupName, Resource: genericapiserver.AllResources},
		"",
		NewSingleContentTypeSerializer(api.Scheme, testapi.Certificates.Codec(), runtime.ContentTypeJSON))

	return &master.Config{
		Config: &genericapiserver.Config{
			StorageFactory:          storageFactory,
			APIResourceConfigSource: master.DefaultAPIResourceConfigSource(),
			APIPrefix:               "/api",
			APIGroupPrefix:          "/apis",
			Authorizer:              apiserver.NewAlwaysAllowAuthorizer(),
			AdmissionControl:        admit.NewAlwaysAdmit(),
			Serializer:              api.Codecs,
			EnableWatchCache:        true,
			// Set those values to avoid annoying warnings in logs.
			ServiceClusterIPRange: parseCIDROrDie("10.0.0.0/24"),
			ServiceNodePortRange:  utilnet.PortRange{Base: 30000, Size: 2768},
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
	masterConfig.APIResourceConfigSource = master.DefaultAPIResourceConfigSource()
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
		m.ApiServer.Close()
	}
}

func CreateTestingNamespace(baseName string, apiserver *httptest.Server, t *testing.T) *api.Namespace {
	// TODO: Create a namespace with a given basename.
	// Currently we neither create the namespace nor delete all its contents at the end.
	// But as long as tests are not using the same namespaces, this should work fine.
	return &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			// TODO: Once we start creating namespaces, switch to GenerateName.
			Name: baseName,
		},
	}
}

func DeleteTestingNamespace(ns *api.Namespace, apiserver *httptest.Server, t *testing.T) {
	// TODO: Remove all resources from a given namespace once we implement CreateTestingNamespace.
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
func ScaleRC(name, ns string, replicas int32, restClient *client.Client) (*api.ReplicationController, error) {
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

// StartPods check for numPods in namespace. If they exist, it no-ops, otherwise it starts up
// a temp rc, scales it to match numPods, then deletes the rc leaving behind the pods.
func StartPods(namespace string, numPods int, host string, restClient *client.Client) error {
	start := time.Now()
	defer func() {
		glog.Infof("StartPods took %v with numPods %d", time.Since(start), numPods)
	}()
	hostField := fields.OneTermEqualSelector(api.PodHostField, host)
	options := api.ListOptions{FieldSelector: hostField}
	pods, err := restClient.Pods(namespace).List(options)
	if err != nil || len(pods.Items) == numPods {
		return err
	}
	glog.Infof("Found %d pods that match host %v, require %d", len(pods.Items), hostField, numPods)
	// For the sake of simplicity, assume all pods in namespace have selectors matching TestRCManifest.
	controller := RCFromManifest(TestRCManifest)

	// Overwrite namespace
	controller.ObjectMeta.Namespace = namespace
	controller.Spec.Template.ObjectMeta.Namespace = namespace

	// Make the rc unique to the given host.
	controller.Spec.Replicas = int32(numPods)
	controller.Spec.Template.Spec.NodeName = host
	controller.Name = controller.Name + host
	controller.Spec.Selector["host"] = host
	controller.Spec.Template.Labels["host"] = host

	if rc, err := StartRC(controller, restClient); err != nil {
		return err
	} else {
		// Delete the rc, otherwise when we restart master components for the next benchmark
		// the rc controller will race with the pods controller in the rc manager.
		return restClient.ReplicationControllers(namespace).Delete(rc.Name, nil)
	}
}

func RunAMaster(masterConfig *master.Config) (*master.Master, *httptest.Server) {
	if masterConfig == nil {
		masterConfig = NewMasterConfig()
		masterConfig.EnableProfiling = true
	}
	return startMasterOrDie(masterConfig)
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
