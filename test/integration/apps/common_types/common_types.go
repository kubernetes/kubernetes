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

package commontypes

import (
	"fmt"
	"os"
	"strconv"
	"syscall"
	"testing"
	"time"

	//"k8s.io/api/core/v1"
	dockertypes "github.com/docker/docker/api/types"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	cadvisortest "k8s.io/kubernetes/pkg/kubelet/cadvisor/testing"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
	"k8s.io/kubernetes/pkg/kubemark"
	"k8s.io/kubernetes/pkg/volume"
	fakevolume "k8s.io/kubernetes/pkg/volume/testing"
)

const (
	//DataDir in which test specific data is entered
	DataDir = "/data"
	//NumNodes Configure numebr of nodes here 10 Hollow Nodes is default
	NumNodes = 10
	//RemoteEndPoint To run it as non-root Overwrite the docker unix-socket location
	RemoteEndPoint = "dockershim.sock"
	//SandboxDir Location pod sandbox info is created
	SandboxDir = "/dockershim"
	//RootDir Root of docker (pod) information per node
	RootDir = "/docker_root"
	//PodManifiestDir Nodes pod manifies location
	PodManifiestDir = RootDir + "/manifiest"
	//MasterPort Known master node
	MasterPort = "8585"
	//NameSpace Default Namespace used by majority of the tests
	//NameSpace = "AppIntegrationTest"
	NameSpace = "default"
	//NodeName Name of the hollow node
	NodeName = "hollow"
	//ConfigFile kubelet's clientset
	ConfigFile = "/config/kubelet.conf"
	//AppTestDir Integration test location
	AppTestDir = "/test/integration/apps"
)

//Config a common config that will be shared across all the tests (sub-tests)
type Config struct {

	//Client information
	Cli *clientset.Clientset

	//Master Related Configs
	Master string

	//A Common Namespace used mostly for Tests
	NameSpace string

	Nodes *HollowNodes

	//RootDir of this test
	RootDir string
}

//Cfg is that one that any tests (sub-test) will have to include this package and can access common.Cfg
var Cfg Config

//A HollowNode and its components, that we may use in our test to verify
type HollowNode struct {
	Name            string                      // Name of the HollowNode
	ConfigFile      string                      // Client config file
	NodePath        string                      // Nodepath (root dir for this node)
	RemoteEndPoint  string                      //redirect Remote endpoint from /var/run/
	RootDir         string                      // Docker root dir
	SandboxDir      string                      // Sandbox Dir for the kubelet
	PodManifestPath string                      // General Pod manifies path
	CadvisorMgr     *cadvisortest.Fake          // CAdvisor Manager
	DockerCli       *libdocker.FakeDockerClient // Fake Docker client if we need to check any
	ContainerMgr    cm.ContainerManager         // Fake Container manager
	VPlugins        []volume.VolumePlugin       // Fake Volume Plugins
	Cli             *clientset.Clientset        // ClientSet
	HollowKubelet   *kubemark.HollowKubelet     // Hallow kube mark

}

func createCli(t *testing.T) *clientset.Clientset {

	basePath := Cfg.RootDir + AppTestDir + DataDir
	configFile := basePath + ConfigFile
	clientConfig, err := clientcmd.LoadFromFile(configFile)
	if err != nil {
		t.Errorf("error while loading kubeconfig from file %v: %v", configFile, err)
	}

	config, err := clientcmd.NewDefaultClientConfig(*clientConfig, &clientcmd.ConfigOverrides{}).ClientConfig()
	if err != nil {
		t.Errorf("error while creating kubeconfig: %v", err)
	}
	config.ContentType = "application/vnd.kubernetes.protobuf"
	config.QPS = 10
	config.Burst = 20

	Cli, err := clientset.NewForConfig(config)
	if err != nil {
		t.Errorf("Failed to create a ClientSet: %v. Exiting.\n", err)
	}

	return Cli

}

//NewHollowNode creates a Hollow node structure
func NewHollowNode(t *testing.T, index int) *HollowNode {
	var hn *HollowNode
	Name := fmt.Sprintf("%s%d", NodeName, index)
	basePath := Cfg.RootDir + AppTestDir + DataDir
	nodePath := fmt.Sprintf("%s/nodes/%s", basePath, Name)
	hn = &HollowNode{Name: Name, NodePath: nodePath}

	hn.ConfigFile = basePath + ConfigFile

	hn.Cli = createCli(t)

	hn.CadvisorMgr = &cadvisortest.Fake{
		NodeName: "testing-node1",
	}

	hn.ContainerMgr = cm.NewStubContainerManager()
	hn.DockerCli = libdocker.NewFakeDockerClient().WithTraceDisabled()
	hn.DockerCli.EnableSleep = true

	hollowKubelet := kubemark.NewHollowKubelet(
		Name,
		hn.Cli,
		hn.CadvisorMgr,
		hn.DockerCli,
		10250+(index*10),
		10500+(index*10),
		hn.ContainerMgr,
		110,
		0,
	)

	hn.RemoteEndPoint = fmt.Sprintf("unix://%s/%s", nodePath, RemoteEndPoint)
	hn.RootDir = fmt.Sprintf("%s/%s/", nodePath, RootDir)
	hn.SandboxDir = fmt.Sprintf("%s/%s", nodePath, SandboxDir)
	hn.PodManifestPath = fmt.Sprintf("%s/%s", nodePath, PodManifiestDir)

	hollowKubelet.KubeletConfiguration.RemoteRuntimeEndpoint = hn.RemoteEndPoint
	hollowKubelet.KubeletFlags.DockershimRootDirectory = hn.RootDir
	hollowKubelet.KubeletConfiguration.PodManifestPath = hn.PodManifestPath
	//hollowKubelet.KubeletConfiguration.RootDirectory = hn.SandboxDir

	_, fkvolumePlugin := fakevolume.GetTestVolumePluginMgr(t)

	volumePlugins := []volume.VolumePlugin{fkvolumePlugin}
	hn.VPlugins = volumePlugins
	hollowKubelet.KubeletDeps.VolumePlugins = hn.VPlugins

	hn.HollowKubelet = hollowKubelet

	return hn
}

func createDIR(t *testing.T, dirname string) {

	err := os.Mkdir(dirname, 0777)
	if err != nil {
		t.Errorf("Unable to create the dir=%s err=%v", dirname, err)
	}
}

//Init just initalize hollow node by creating necessary work directories
func (hn *HollowNode) Init(t *testing.T) {

	//Create the directories for this hollow node to operate upon.
	//Create the directories for this hollow node to operate upon.
	createDIR(t, hn.NodePath)
	createDIR(t, hn.RootDir)
	createDIR(t, hn.PodManifestPath)
	createDIR(t, hn.SandboxDir)

}

//Run simply call holow nodes Run() method in a go-routine
func (hn *HollowNode) Run() {

	go hn.HollowKubelet.Run()
	time.Sleep(time.Second)
}

//HollowNodes Map of nodes we might need to lookup in the tests.
type HollowNodes struct {
	N map[string]*HollowNode //A Map of hollow nodes

}

//Add simply adds the hollownode to the lookup map
func (hns *HollowNodes) Add(hn *HollowNode) {
	hns.N[hn.Name] = hn
}

// ListContainers ListContainers prints all the containers from the fake Docker client from all the nodes
func (hns *HollowNodes) ListContainers() []string {

	var result []string
	var opt dockertypes.ContainerListOptions
	opt.All = true

	for _, n := range hns.N {
		containers, _ := n.DockerCli.ListContainers(opt)
		for _, c := range containers {
			result = append(result, n.Name+":"+c.ID)
		}
	}

	return result
}

// Initialize Initalize test suite to do the following
// 1) Create NumNodes * Hollow-Nodes
// 2) Monitor Kube-system processes such as ApiServer, Controller Manager and Scheduler
func Initialize(t *testing.T) error {

	// Get the root dir
	Cfg.RootDir = os.Getenv("KUBE_ROOT")
	Cfg.Master = fmt.Sprintf("http://localhost:%s", MasterPort)
	Cfg.NameSpace = NameSpace
	Cfg.Nodes = &HollowNodes{N: make(map[string]*HollowNode)}
	Cfg.Cli = createCli(t)
	/*
		var ns v1.Namespace
		ns.ObjectMeta.Name = Cfg.NameSpace
		_, err := Cfg.Cli.Core().Namespaces().Create(&ns)
		if err != nil {
			t.Errorf("Error creating namespaces err=%v", err)
		}
	*/
	fmt.Printf("Creating %d Hollow-nodes...\n", NumNodes)

	for i := 0; i < NumNodes; i++ {

		hn := NewHollowNode(t, i)
		hn.Init(t)
		Cfg.Nodes.Add(hn)
		hn.Run()
	}

	fmt.Printf("Monitoring and waiting for kubernetes components..\n")
	waitForKubePIDs()

	return nil

}

// Utility functions
func getPID(env string) int {

	envVar := os.Getenv(env)
	if envVar == "" {
		return -1
	}
	pid, err := strconv.Atoi(envVar)
	if err != nil {
		return -1
	}
	return pid
}

func waitForPID(name string, env string) error {

	pid := getPID(env)
	if pid == -1 {
		fmt.Printf("Looks like %s is not started invalid pid=%s\n", name, env)
		os.Exit(1)
	}

	proc, _ := os.FindProcess(pid)
	for {
		err := proc.Signal(syscall.Signal(0))
		if err != nil {
			fmt.Printf("Looks like %s exited prematurely err=%v\n", name, err)
			os.Exit(1)
		}
		time.Sleep(time.Second)
	}
}

// CheckErrors CheckErrors will mark the test as fail if there err value is not null.
func CheckErrors(t *testing.T, err error, format string, args ...interface{}) {
	if err != nil {
		t.Errorf("Error Occured err=%v msg=%s", err, fmt.Sprintf(format, args...))
	}
}

// We should keep monitoring API Server, Scheduler and Controller Manager
// If one of them crashed during the testing, there is no point in
// continuing.
func waitForKubePIDs() {

	fmt.Printf("Waiting for Kuberentes Components..\n")

	go waitForPID("API Server", "API_SRV_PID")
	go waitForPID("Scheduler", "SCHED_PID")
	go waitForPID("Controller Manager", "CM_PID")

}
