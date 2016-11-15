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
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/apiserver/authenticator"
	authauthenticator "k8s.io/kubernetes/pkg/auth/authenticator"
	authauthorizer "k8s.io/kubernetes/pkg/auth/authorizer"
	authorizerunion "k8s.io/kubernetes/pkg/auth/authorizer/union"
	"k8s.io/kubernetes/pkg/auth/user"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	coreclient "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/core/v1"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/controller"
	replicationcontroller "k8s.io/kubernetes/pkg/controller/replication"
	"k8s.io/kubernetes/pkg/generated/openapi"
	"k8s.io/kubernetes/pkg/genericapiserver"
	"k8s.io/kubernetes/pkg/genericapiserver/authorizer"
	"k8s.io/kubernetes/pkg/kubectl"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage/storagebackend"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/version"
	"k8s.io/kubernetes/pkg/watch"
	"k8s.io/kubernetes/plugin/pkg/admission/admit"
	authenticatorunion "k8s.io/kubernetes/plugin/pkg/auth/authenticator/request/union"

	"github.com/go-openapi/spec"
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
	ClientSet clientset.Interface
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
	m, s := startMasterOrDie(c.MasterConfig, nil, nil)
	// TODO: Allow callers to pipe through a different master url and create a client/start components using it.
	glog.Infof("Master %+v", s.URL)
	// TODO: caesarxuchao: remove this client when the refactoring of client libraray is done.
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &registered.GroupOrDie(api.GroupName).GroupVersion}, QPS: c.QPS, Burst: c.Burst})
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
		ClientSet:         clientset,
		ControllerManager: controllerManager,
		rcStopCh:          rcStopCh,
	}
}

// alwaysAllow always allows an action
type alwaysAllow struct{}

func (alwaysAllow) Authorize(requestAttributes authauthorizer.Attributes) (bool, string, error) {
	return true, "always allow", nil
}

// alwaysEmpty simulates "no authentication" for old tests
func alwaysEmpty(req *http.Request) (user.Info, bool, error) {
	return &user.DefaultInfo{
		Name: "",
	}, true, nil
}

// MasterReceiver can be used to provide the master to a custom incoming server function
type MasterReceiver interface {
	SetMaster(m *master.Master)
}

// MasterHolder implements
type MasterHolder struct {
	Initialized chan struct{}
	M           *master.Master
}

func (h *MasterHolder) SetMaster(m *master.Master) {
	h.M = m
	close(h.Initialized)
}

// startMasterOrDie starts a kubernetes master and an httpserver to handle api requests
func startMasterOrDie(masterConfig *master.Config, incomingServer *httptest.Server, masterReceiver MasterReceiver) (*master.Master, *httptest.Server) {
	var m *master.Master
	var s *httptest.Server

	if incomingServer != nil {
		s = incomingServer
	} else {
		s = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			m.GenericAPIServer.Handler.ServeHTTP(w, req)
		}))
	}

	if masterConfig == nil {
		masterConfig = NewMasterConfig()
		masterConfig.GenericConfig.EnableProfiling = true
		masterConfig.GenericConfig.EnableMetrics = true
		masterConfig.GenericConfig.EnableSwaggerSupport = true
		masterConfig.GenericConfig.EnableOpenAPISupport = true
		masterConfig.GenericConfig.OpenAPIConfig.Info = &spec.Info{
			InfoProps: spec.InfoProps{
				Title:   "Kubernetes",
				Version: "unversioned",
			},
		}
		masterConfig.GenericConfig.OpenAPIConfig.DefaultResponse = &spec.Response{
			ResponseProps: spec.ResponseProps{
				Description: "Default Response.",
			},
		}
		masterConfig.GenericConfig.OpenAPIConfig.Definitions = openapi.OpenAPIDefinitions
	}

	// set the loopback client config
	if masterConfig.GenericConfig.LoopbackClientConfig == nil {
		masterConfig.GenericConfig.LoopbackClientConfig = &restclient.Config{QPS: 50, Burst: 100, ContentConfig: restclient.ContentConfig{NegotiatedSerializer: api.Codecs}}
	}
	masterConfig.GenericConfig.LoopbackClientConfig.Host = s.URL

	privilegedLoopbackToken := uuid.NewRandom().String()
	// wrap any available authorizer
	tokens := make(map[string]*user.DefaultInfo)
	tokens[privilegedLoopbackToken] = &user.DefaultInfo{
		Name:   user.APIServerUser,
		UID:    uuid.NewRandom().String(),
		Groups: []string{user.SystemPrivilegedGroup},
	}

	tokenAuthenticator := authenticator.NewAuthenticatorFromTokens(tokens)
	if masterConfig.GenericConfig.Authenticator == nil {
		masterConfig.GenericConfig.Authenticator = authenticatorunion.New(tokenAuthenticator, authauthenticator.RequestFunc(alwaysEmpty))
	} else {
		masterConfig.GenericConfig.Authenticator = authenticatorunion.New(tokenAuthenticator, masterConfig.GenericConfig.Authenticator)
	}

	if masterConfig.GenericConfig.Authorizer != nil {
		tokenAuthorizer := authorizer.NewPrivilegedGroups(user.SystemPrivilegedGroup)
		masterConfig.GenericConfig.Authorizer = authorizerunion.New(tokenAuthorizer, masterConfig.GenericConfig.Authorizer)
	} else {
		masterConfig.GenericConfig.Authorizer = alwaysAllow{}
	}

	masterConfig.GenericConfig.LoopbackClientConfig.BearerToken = privilegedLoopbackToken

	m, err := masterConfig.Complete().New()
	if err != nil {
		glog.Fatalf("error in bringing up the master: %v", err)
	}
	if masterReceiver != nil {
		masterReceiver.SetMaster(m)
	}

	// TODO have this start method actually use the normal start sequence for the API server
	// this method never actually calls the `Run` method for the API server
	// fire the post hooks ourselves
	m.GenericAPIServer.PrepareRun()
	m.GenericAPIServer.RunPostStartHooks()

	cfg := *masterConfig.GenericConfig.LoopbackClientConfig
	cfg.ContentConfig.GroupVersion = &unversioned.GroupVersion{}
	privilegedClient, err := restclient.RESTClientFor(&cfg)
	if err != nil {
		glog.Fatal(err)
	}
	err = wait.PollImmediate(100*time.Millisecond, 30*time.Second, func() (bool, error) {
		result := privilegedClient.Get().AbsPath("/healthz").Do()
		status := 0
		result.StatusCode(&status)
		if status == 200 {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		glog.Fatal(err)
	}

	// wait for services to be ready
	if masterConfig.EnableCoreControllers {
		// TODO Once /healthz is updated for posthooks, we'll wait for good health
		coreClient := coreclient.NewForConfigOrDie(&cfg)
		svcWatch, err := coreClient.Services(api.NamespaceDefault).Watch(v1.ListOptions{})
		if err != nil {
			glog.Fatal(err)
		}
		_, err = watch.Until(30*time.Second, svcWatch, func(event watch.Event) (bool, error) {
			if event.Type != watch.Added {
				return false, nil
			}
			if event.Object.(*v1.Service).Name == "kubernetes" {
				return true, nil
			}
			return false, nil
		})
		if err != nil {
			glog.Fatal(err)
		}
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

	info, _ := runtime.SerializerInfoForMediaType(api.Codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
	ns := NewSingleContentTypeSerializer(api.Scheme, info)

	storageFactory := genericapiserver.NewDefaultStorageFactory(config, runtime.ContentTypeJSON, ns, genericapiserver.NewDefaultResourceEncodingConfig(), master.DefaultAPIResourceConfigSource())
	storageFactory.SetSerializer(
		unversioned.GroupResource{Group: api.GroupName, Resource: genericapiserver.AllResources},
		"",
		ns)
	storageFactory.SetSerializer(
		unversioned.GroupResource{Group: autoscaling.GroupName, Resource: genericapiserver.AllResources},
		"",
		ns)
	storageFactory.SetSerializer(
		unversioned.GroupResource{Group: batch.GroupName, Resource: genericapiserver.AllResources},
		"",
		ns)
	storageFactory.SetSerializer(
		unversioned.GroupResource{Group: apps.GroupName, Resource: genericapiserver.AllResources},
		"",
		ns)
	storageFactory.SetSerializer(
		unversioned.GroupResource{Group: extensions.GroupName, Resource: genericapiserver.AllResources},
		"",
		ns)
	storageFactory.SetSerializer(
		unversioned.GroupResource{Group: policy.GroupName, Resource: genericapiserver.AllResources},
		"",
		ns)
	storageFactory.SetSerializer(
		unversioned.GroupResource{Group: rbac.GroupName, Resource: genericapiserver.AllResources},
		"",
		ns)
	storageFactory.SetSerializer(
		unversioned.GroupResource{Group: certificates.GroupName, Resource: genericapiserver.AllResources},
		"",
		ns)
	storageFactory.SetSerializer(
		unversioned.GroupResource{Group: storage.GroupName, Resource: genericapiserver.AllResources},
		"",
		ns)

	genericConfig := genericapiserver.NewConfig()
	kubeVersion := version.Get()
	genericConfig.Version = &kubeVersion
	genericConfig.APIResourceConfigSource = master.DefaultAPIResourceConfigSource()
	genericConfig.Authorizer = authorizer.NewAlwaysAllowAuthorizer()
	genericConfig.AdmissionControl = admit.NewAlwaysAdmit()
	genericConfig.EnableMetrics = true

	return &master.Config{
		GenericConfig:         genericConfig,
		StorageFactory:        storageFactory,
		EnableCoreControllers: true,
		EnableWatchCache:      true,
		KubeletClientConfig:   kubeletclient.KubeletClientConfig{Port: 10250},
		APIServerServicePort:  443,
		MasterCount:           1,
	}
}

// Returns the master config appropriate for most integration tests.
func NewIntegrationTestMasterConfig() *master.Config {
	masterConfig := NewMasterConfig()
	masterConfig.EnableCoreControllers = true
	masterConfig.GenericConfig.PublicAddress = net.ParseIP("192.168.10.4")
	masterConfig.GenericConfig.APIResourceConfigSource = master.DefaultAPIResourceConfigSource()
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
func StopRC(rc *api.ReplicationController, clientset clientset.Interface) error {
	reaper, err := kubectl.ReaperFor(api.Kind("ReplicationController"), clientset)
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
func ScaleRC(name, ns string, replicas int32, clientset clientset.Interface) (*api.ReplicationController, error) {
	scaler, err := kubectl.ScalerFor(api.Kind("ReplicationController"), clientset)
	if err != nil {
		return nil, err
	}
	retry := &kubectl.RetryParams{Interval: 50 * time.Millisecond, Timeout: DefaultTimeout}
	waitForReplicas := &kubectl.RetryParams{Interval: 50 * time.Millisecond, Timeout: DefaultTimeout}
	err = scaler.Scale(ns, name, uint(replicas), nil, retry, waitForReplicas)
	if err != nil {
		return nil, err
	}
	scaled, err := clientset.Core().ReplicationControllers(ns).Get(name)
	if err != nil {
		return nil, err
	}
	return scaled, nil
}

func RunAMaster(masterConfig *master.Config) (*master.Master, *httptest.Server) {
	if masterConfig == nil {
		masterConfig = NewMasterConfig()
		masterConfig.GenericConfig.EnableProfiling = true
		masterConfig.GenericConfig.EnableMetrics = true
	}
	return startMasterOrDie(masterConfig, nil, nil)
}

func RunAMasterUsingServer(masterConfig *master.Config, s *httptest.Server, masterReceiver MasterReceiver) (*master.Master, *httptest.Server) {
	return startMasterOrDie(masterConfig, s, masterReceiver)
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
