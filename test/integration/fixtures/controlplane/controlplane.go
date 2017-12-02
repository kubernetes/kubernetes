/*
Copyright 2017 The Kubernetes Authors.

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

package controlplane

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"sync"
	"testing"
	"time"

	etcdtesting "k8s.io/apiserver/pkg/storage/etcd/testing"

	ApiTesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	cmapp "k8s.io/kubernetes/cmd/kube-controller-manager/app"
	cmopt "k8s.io/kubernetes/cmd/kube-controller-manager/app/options"
	sched "k8s.io/kubernetes/cmd/kube-scheduler/app"

	hwnodes "k8s.io/kubernetes/test/integration/fixtures/hollownode"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	clientcmd "k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/kubernetes/pkg/util/configz"

	"github.com/coreos/pkg/capnslog"
)

//KubeConfFileName The filename kubeconfig file is created with
const KubeConfFileName = "kubeconfig"

//Declare Some Global variables
var (
	//RegisteredControllPlanes Map of all registred control plances
	RegisteredControllPlanes map[string]*ControlPlane //Try to maintain a map of controll planes
	//InitOnce Do.Once for Initializing this map
	InitOnce sync.Once //Initialize this map only once
)

//APIServer Structure of the API server object
type APIServer struct {
	ApiTesting.TestServer
	InsecurePort int
	etcdServer   *etcdtesting.EtcdTestServer
}

//Start Starts the API serer, should be fatal on any errors
func (AP *APIServer) Start(t *testing.T) error {

	var flags []string
	var err error

	storageServer, storageConfig := etcdtesting.NewUnsecuredEtcd3TestClientServer(t)

	//Supress any futher logs for etcd server
	repo, err := capnslog.GetRepoLogger("github.com/coreos/etcd")
	if err != nil {
		t.Fatalf("couldn't configure logging: %v", err)
	}
	repo.SetRepoLogLevel(capnslog.ERROR)

	//Supply insecure port
	AP.InsecurePort, err = createListenerOnFreePort()
	if err != nil {
		t.Errorf("Unable to obtain an insecure port=%v\n", err)
		return nil
	}

	flags = append(flags, fmt.Sprintf("--insecure-port=%d", AP.InsecurePort))

	//Supply plugins
	flags = append(flags, "--admission-control=Initializers,NamespaceLifecycle,LimitRanger,DefaultStorageClass,ResourceQuota,DefaultTolerationSeconds")

	testAPIServer := ApiTesting.StartTestServerOrDie(t, flags, storageConfig)

	AP.ClientConfig = testAPIServer.ClientConfig
	AP.ServerOpts = testAPIServer.ServerOpts
	AP.TearDownFn = testAPIServer.TearDownFn
	AP.TmpDir = testAPIServer.TmpDir
	AP.etcdServer = storageServer

	return nil
}

//TearDown Brings down an api-server gracefully.
func (AP *APIServer) TearDown(t *testing.T) error {

	AP.TearDownFn()
	AP.etcdServer.Terminate(t)

	return nil

}

//ControllerManager a simple struct for CM maintenance
type ControllerManager struct {
	Opt  *cmopt.CMServer
	Port int
}

//Start Starts a Controlelr Manager
func (CM *ControllerManager) Start(t *testing.T, cli *restclient.Config, KubeConfigFilePath string) error {

	var err error

	CM.Opt = cmopt.NewCMServer()

	CM.Opt.Kubeconfig = KubeConfigFilePath
	CM.Port, err = createListenerOnFreePort()
	if err != nil {
		t.Fatalf("Failed to create listerner port for Controlelr Manager err=%v\n", err)
	}
	CM.Opt.Port = int32(CM.Port)
	CM.Opt.Controllers = append(CM.Opt.Controllers, "-nodelifecycle")
	CM.Opt.Controllers = append(CM.Opt.Controllers, "-serviceaccount")
	CM.Opt.Controllers = append(CM.Opt.Controllers, "-bootstrapsigner")
	CM.Opt.Controllers = append(CM.Opt.Controllers, "-tokencleaner")
	CM.Opt.Controllers = append(CM.Opt.Controllers, cmapp.KnownControllers()...)

	CM.Opt.UseServiceAccountCredentials = false

	go func(t *testing.T) {

		err = cmapp.Run(CM.Opt)
		if err != nil {
			t.Fatalf("Error Starting Controller manager error = %v\n", err)
		}
	}(t)

	log.Printf("Waiting for the controlelr manager to come up Port=%d CM.Port=%d", CM.Port, CM.Opt.Port)

	url := fmt.Sprintf("http://0.0.0.0:%d/healthz", CM.Port)

	//Wait until the controller manafer is up
	for {
		log.Printf("Tring to http.Get url=%s", url)
		time.Sleep(time.Millisecond * 100)
		if Healthz(url) {
			break
		}
	}

	t.Logf("Controlelr manager started...\n")

	return err
}

//Scheduler a Simple struct for Scheduler maintenance
type Scheduler struct {
	Opt  *sched.Options
	Port int
}

//Start starts a scheduler
func (Sc *Scheduler) Start(t *testing.T, cli *restclient.Config, KubeConfigFilePath string) error {

	var err error
	var flags []string

	command := sched.NewSchedulerCommand()

	Sc.Port, err = createListenerOnFreePort()
	if err != nil {
		t.Errorf("Unable to create Lister port for scheduler=%v\n", err)
	}

	flags = append(flags, "--kubeconfig="+KubeConfigFilePath)
	flags = append(flags, fmt.Sprintf("--port=%d", Sc.Port))

	command.SetArgs(flags)

	go func(t *testing.T) {
		err = command.Execute()
		if err != nil {
			t.Fatalf("Unable to start Scheduler failed =%v\n", err)
		}
	}(t)

	log.Printf("Waiting for the Scheduler to come up Port=%d", Sc.Port)
	//Wait until the scheduler is up
	for {
		time.Sleep(time.Millisecond * 100)
		if Healthz(fmt.Sprintf("http://0.0.0.0:%d/healthz", Sc.Port)) {
			break
		}
	}
	t.Logf("Scheduler started...\n")
	return nil
}

//ControlPlane Identifiles starts a control plane
type ControlPlane struct {
	Name              string
	APIServer         APIServer
	ControllerManager ControllerManager
	Scheduler         Scheduler
	Cli               *kubernetes.Clientset
	Conf              *clientcmdapi.Config
}

//Start Starts complete control plane
func (CP *ControlPlane) Start(t *testing.T) error {

	CP.APIServer.Start(t)
	CP.Cli, _ = kubernetes.NewForConfig(CP.APIServer.ClientConfig)

	certPem, keyPem, err := certutil.GenerateSelfSignedCertKey(server.LoopbackClientServerNameOverride, nil, nil)
	if err != nil {
		t.Fatalf("failed to generate self-signed certificate for loopback connection: %v", err)
	}

	CP.Conf = clientcmdapi.NewConfig()
	Cluster := clientcmdapi.NewCluster()
	Context := clientcmdapi.NewContext()
	AuthInfo := clientcmdapi.NewAuthInfo()

	//API-Server
	KubeConfigFilePath := fmt.Sprintf("%s/%s", CP.APIServer.TmpDir, KubeConfFileName)

	Cluster.Server = CP.APIServer.ClientConfig.Host
	Cluster.InsecureSkipTLSVerify = true

	Context.AuthInfo = CP.Name
	Context.Cluster = CP.Name

	AuthInfo.Token = CP.APIServer.ClientConfig.BearerToken
	AuthInfo.ClientKeyData = append(AuthInfo.ClientKeyData, keyPem...)
	AuthInfo.ClientCertificateData = append(AuthInfo.ClientCertificateData, certPem...)

	CP.Conf.CurrentContext = CP.Name
	CP.Conf.Clusters[CP.Name] = Cluster
	CP.Conf.Contexts[CP.Name] = Context
	CP.Conf.AuthInfos[CP.Name] = AuthInfo

	err = clientcmd.WriteToFile(*CP.Conf, KubeConfigFilePath)
	if err != nil {
		t.Fatalf("Unable to write the config locally")
	}

	//Create an output file
	logFile, err := ioutil.TempFile(CP.APIServer.TmpDir, CP.Name)
	log.SetOutput(logFile)
	flag.Lookup("log_dir").Value.Set(CP.APIServer.TmpDir)

	CP.ControllerManager.Start(t, CP.APIServer.ClientConfig, KubeConfigFilePath)
	configz.Delete("componentconfig")
	CP.Scheduler.Start(t, CP.APIServer.ClientConfig, KubeConfigFilePath)

	//Add nodes to the cluster. Default is ten nodes
	CP.AddNode(t)

	RegisterControllPlane(CP.Name, CP)

	return nil
}

//HealthCheck Api server will healthcheck itself just do it for Controller Manager and Scheduler
func (CP *ControlPlane) HealthCheck() {

	//We need to abort if any of the component is failing health check

	err := wait.Poll(500*time.Millisecond, 30*time.Second, func() (bool, error) {

		if !Healthz(fmt.Sprintf("http://0.0.0.0:%d/healthz", CP.ControllerManager.Port)) {
			log.Fatalf("Controller Manager failed health-check aborting")

		}

		if !Healthz(fmt.Sprintf("http://0.0.0.0:%d/healthz", CP.Scheduler.Port)) {
			log.Fatalf("Scheduler failed health-check aborting")
		}

		log.Printf("Controller Manager 0.0.0.0:%d and Scheduler 0.0.0.0:%d are healthy", CP.ControllerManager.Port, CP.Scheduler.Port)

		return true, nil

	})
	if err != nil {

		log.Fatalf("Unable to wait for healthcheck\n")
	}

}

//AddNode Addss a number of nodes to the control plane, for now they are hollow nodes
func (CP *ControlPlane) AddNode(t *testing.T) {

	hwnodes.InitNodes(t, CP.APIServer.TmpDir, CP.APIServer.InsecurePort)
}

//Healthz Health checks a given url, just verifies if it is 200 OK.
func Healthz(url string) bool {
	res, err := http.Get(url)
	if err == nil && res.StatusCode == http.StatusOK {
		return true
	}
	return false
}

//TearDown bring down a control plane gracefully.
func (CP *ControlPlane) TearDown(t *testing.T) error {

	//Teardown API Server
	CP.APIServer.TearDown(t)
	return nil
}

//NewControlPlane creates a new pointer of CP identified by a given name
func NewControlPlane(name string) *ControlPlane {
	return &ControlPlane{Name: name}
}

//createListenerOnFreePort a local function to pick a random port to be used by the component to bind.
func createListenerOnFreePort() (int, error) {

	ln, err := net.Listen("tcp", ":0")
	if err != nil {
		return 0, err
	}

	//Close this anyways as we will try to bind again.
	defer ln.Close()

	// get port
	tcpAddr, ok := ln.Addr().(*net.TCPAddr)
	if !ok {
		return 0, fmt.Errorf("invalid listen address: %q", ln.Addr().String())
	}
	return tcpAddr.Port, nil

}

//LookupControllPlane a top-level funtion to simply check if a mentioned control plane exisist
func LookupControllPlane(name string) *ControlPlane {

	if cp, isAvailable := RegisteredControllPlanes[name]; isAvailable {
		return cp
	}

	return nil
}

//RegisterControllPlane Should be automatic as and when a control plance is created it should be registered here.
func RegisterControllPlane(name string, cp *ControlPlane) {

	RegisteredControllPlanes[name] = cp
	return
}

//
func initializeModule() {
	RegisteredControllPlanes = make(map[string]*ControlPlane)
}

func init() {
	InitOnce.Do(initializeModule)
}

//CheckErrors a simple utility function that Fatal's out in case of any error.
func CheckErrors(t *testing.T, err error, msg string) {

	if err == nil {
		return
	}
	t.Fatalf("Error %v occured msg=%s", err, msg)
}
