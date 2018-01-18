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

package controlplane

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"testing"
	"time"

	etcdtesting "k8s.io/apiserver/pkg/storage/etcd/testing"

	apitesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	cmapp "k8s.io/kubernetes/cmd/kube-controller-manager/app"
	cmopt "k8s.io/kubernetes/cmd/kube-controller-manager/app/options"
	sched "k8s.io/kubernetes/cmd/kube-scheduler/app"

	hwnodes "k8s.io/kubernetes/test/integration/fixtures/hollownode"
	"k8s.io/kubernetes/test/integration/framework"

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

//Tests creates a common type for creating and adding tests using this framework
type Tests struct {
	Name string
	F    func(t *testing.T, controlPlane *ControlPlane)
}

//APIServer Structure of the API server object
type APIServer struct {
	apitesting.TestServer
	InsecurePort int
	etcdServer   *etcdtesting.EtcdTestServer
}

//Start Starts the API server, should be fatal on any errors
func (ap *APIServer) Start(t *testing.T) {

	var err error

	storageServer, storageConfig := etcdtesting.NewUnsecuredEtcd3TestClientServer(t)

	//Suppress any further logs for etcd server
	repo, err := capnslog.GetRepoLogger("github.com/coreos/etcd")
	if err != nil {
		t.Fatalf("couldn't configure logging: %v", err)
	}
	repo.SetRepoLogLevel(capnslog.ERROR)

	//Supply insecure port
	ap.InsecurePort, err = framework.FindFreeLocalPort()
	if err != nil {
		t.Errorf("Unable to obtain an insecure port=%v\n", err)
	}

	var flags []string
	//Supply plugins
	flags = append(flags, fmt.Sprintf("--insecure-port=%d", ap.InsecurePort))
	flags = append(flags, "--admission-control=Initializers,NamespaceLifecycle,LimitRanger,DefaultStorageClass,ResourceQuota,DefaultTolerationSeconds")

	testAPIServer := apitesting.StartTestServerOrDie(t, flags, storageConfig)

	ap.ClientConfig = testAPIServer.ClientConfig
	ap.ServerOpts = testAPIServer.ServerOpts
	ap.TearDownFn = testAPIServer.TearDownFn
	ap.TmpDir = testAPIServer.TmpDir
	ap.etcdServer = storageServer

}

//TearDown Brings down an api-server gracefully.
func (ap *APIServer) TearDown(t *testing.T) error {

	ap.TearDownFn()
	ap.etcdServer.Terminate(t)

	return nil

}

//ControllerManager a simple struct for CM maintenance
type ControllerManager struct {
	Opt  *cmopt.CMServer
	Port int
}

//Start starts a Controller Manager
func (cm *ControllerManager) Start(t *testing.T, cli *restclient.Config, KubeConfigFilePath string) error {

	var err error

	cm.Opt = cmopt.NewCMServer()

	cm.Opt.Kubeconfig = KubeConfigFilePath
	cm.Port, err = framework.FindFreeLocalPort()
	if err != nil {
		t.Fatalf("Failed to create listerner port for Controlelr Manager err=%v\n", err)
	}
	cm.Opt.Port = int32(cm.Port)
	cm.Opt.Controllers = append(cm.Opt.Controllers, "-nodelifecycle")
	cm.Opt.Controllers = append(cm.Opt.Controllers, "-serviceaccount")
	cm.Opt.Controllers = append(cm.Opt.Controllers, "-bootstrapsigner")
	cm.Opt.Controllers = append(cm.Opt.Controllers, "-tokencleaner")
	cm.Opt.Controllers = append(cm.Opt.Controllers, cmapp.KnownControllers()...)

	cm.Opt.UseServiceAccountCredentials = false

	go func(t *testing.T) {

		err = cmapp.Run(cm.Opt)
		if err != nil {
			t.Fatalf("Error Starting Controller manager error = %v\n", err)
		}
	}(t)

	t.Logf("Waiting for the controller manager to come up Port=%d CM.Port=%d", cm.Port, cm.Opt.Port)

	url := fmt.Sprintf("http://0.0.0.0:%d/healthz", cm.Port)

	//Wait until the controller manager is up
	err = wait.Poll(time.Millisecond*100, time.Minute, func() (bool, error) {
		time.Sleep(time.Millisecond * 100)
		if Healthz(url) {
			return true, nil
		}
		return false, nil
	})

	if err != nil {
		t.Fatalf("Control Manager failed to start started=%v\n", err)
	}

	t.Logf("Controller manager started...\n")

	return err
}

//Scheduler a Simple struct for Scheduler maintenance
type Scheduler struct {
	Opt  *sched.Options
	Port int
}

//Start starts a scheduler
func (sc *Scheduler) Start(t *testing.T, cli *restclient.Config, KubeConfigFilePath string) error {

	var err error
	var flags []string

	command := sched.NewSchedulerCommand()

	sc.Port, err = framework.FindFreeLocalPort()
	if err != nil {
		t.Errorf("Unable to create Lister port for scheduler=%v\n", err)
	}

	flags = append(flags, "--kubeconfig="+KubeConfigFilePath)
	flags = append(flags, fmt.Sprintf("--port=%d", sc.Port))

	command.SetArgs(flags)

	go func(t *testing.T) {
		err = command.Execute()
		if err != nil {
			t.Fatalf("Unable to start Scheduler failed =%v\n", err)
		}
	}(t)

	t.Logf("Waiting for the Scheduler to come up Port=%d", sc.Port)
	//Wait until the scheduler is up
	err = wait.Poll(time.Millisecond*100, time.Minute, func() (bool, error) {
		if Healthz(fmt.Sprintf("http://0.0.0.0:%d/healthz", sc.Port)) {
			return true, nil
		}
		return false, nil
	})

	if err != nil {
		t.Fatalf("Unable to start Scheduler error=%v", err)
	}

	t.Logf("Scheduler started...\n")
	return nil
}

//ControlPlane starts a control plane
type ControlPlane struct {
	Name              string
	APIServer         APIServer
	ControllerManager ControllerManager
	Scheduler         Scheduler
	Client            *kubernetes.Clientset
	Conf              *clientcmdapi.Config
}

//Start Starts complete control plane
func (cp *ControlPlane) Start(t *testing.T) error {

	cp.APIServer.Start(t)
	cp.Client, _ = kubernetes.NewForConfig(cp.APIServer.ClientConfig)

	certPem, keyPem, err := certutil.GenerateSelfSignedCertKey(server.LoopbackClientServerNameOverride, nil, nil)
	if err != nil {
		t.Fatalf("failed to generate self-signed certificate for loopback connection: %v", err)
	}

	cp.Conf = clientcmdapi.NewConfig()
	Cluster := clientcmdapi.NewCluster()
	Context := clientcmdapi.NewContext()
	AuthInfo := clientcmdapi.NewAuthInfo()

	//API-Server
	KubeConfigFilePath := fmt.Sprintf("%s/%s", cp.APIServer.TmpDir, KubeConfFileName)

	Cluster.Server = cp.APIServer.ClientConfig.Host
	Cluster.InsecureSkipTLSVerify = true

	Context.AuthInfo = cp.Name
	Context.Cluster = cp.Name

	AuthInfo.Token = cp.APIServer.ClientConfig.BearerToken
	AuthInfo.ClientKeyData = append(AuthInfo.ClientKeyData, keyPem...)
	AuthInfo.ClientCertificateData = append(AuthInfo.ClientCertificateData, certPem...)

	cp.Conf.CurrentContext = cp.Name
	cp.Conf.Clusters[cp.Name] = Cluster
	cp.Conf.Contexts[cp.Name] = Context
	cp.Conf.AuthInfos[cp.Name] = AuthInfo

	err = clientcmd.WriteToFile(*cp.Conf, KubeConfigFilePath)
	if err != nil {
		t.Fatalf("Unable to write the config locally")
	}

	//Create an output file
	logFile, err := ioutil.TempFile(cp.APIServer.TmpDir, cp.Name)
	log.SetOutput(logFile)
	flag.Lookup("log_dir").Value.Set(cp.APIServer.TmpDir)

	cp.ControllerManager.Start(t, cp.APIServer.ClientConfig, KubeConfigFilePath)
	configz.Delete("componentconfig")
	cp.Scheduler.Start(t, cp.APIServer.ClientConfig, KubeConfigFilePath)

	//Add nodes to the cluster. Default is ten nodes
	cp.AddNode(t)

	//Start Monitoring the control plane by constantly health checking them
	go cp.HealthCheck(t)

	return nil
}

//HealthCheck Api server will healthcheck itself just do it for Controller Manager and Scheduler
func (cp *ControlPlane) HealthCheck(t *testing.T) {

	//We need to abort if any of the component is failing health check

	healthFn := func() {

		if !Healthz(fmt.Sprintf("http://0.0.0.0:%d/healthz", cp.ControllerManager.Port)) {
			t.Fatalf("Controller Manager failed health-check aborting")

		}

		if !Healthz(fmt.Sprintf("http://0.0.0.0:%d/healthz", cp.Scheduler.Port)) {
			t.Fatalf("Scheduler failed health-check aborting")
		}

	}
	wait.Forever(healthFn, time.Second)
}

//AddNode Addss a number of nodes to the control plane, for now they are hollow nodes
func (cp *ControlPlane) AddNode(t *testing.T) {

	hwnodes.InitNodes(t, cp.APIServer.TmpDir, cp.APIServer.InsecurePort)
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
func (cp *ControlPlane) TearDown(t *testing.T) error {

	//Teardown API Server
	cp.APIServer.TearDown(t)
	return nil
}

//NewControlPlane creates a new pointer of CP identified by a given name
func NewControlPlane(name string) *ControlPlane {
	return &ControlPlane{Name: name}
}

//CheckErrors a simple utility function that Fatal's out in case of any error.
func CheckErrors(t *testing.T, err error, msg string) {

	if err == nil {
		return
	}
	t.Fatalf("Error %v occured msg=%s", err, msg)
}
