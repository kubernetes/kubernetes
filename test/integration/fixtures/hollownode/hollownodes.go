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

package hollownodes

import (
	"fmt"
	"os"
	"testing"
	"time"

	//"k8s.io/api/core/v1"
	dockertypes "github.com/docker/docker/api/types"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	cadvisortest "k8s.io/kubernetes/pkg/kubelet/cadvisor/testing"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/dockershim"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
	"k8s.io/kubernetes/pkg/kubemark"
	"k8s.io/kubernetes/pkg/volume"
	fakevolume "k8s.io/kubernetes/pkg/volume/testing"
)

const (
	//NumNodes Configure numebr of nodes here 10 Hollow Nodes is default
	NumNodes = 10
	//RemoteEndPoint To run it as non-root Overwrite the docker unix-socket location
	RemoteEndPoint = "dockershim.sock"
	//RootDir Root of docker (pod) information per node
	RootDir = ""
	//NameSpace = "AppIntegrationTest"
	NameSpace = "default"
	//NodeName Name of the hollow node
	NodeName = "hollow"
	//ConfigFile kubelet's clientset
	ConfigFile = "/kubeconfig"
)

//Config a common config that will be shared across all the tests (sub-tests)
type Config struct {

	//Client information
	Client *clientset.Clientset

	//Master Related Configs
	Master string

	//A Common Namespace used mostly for Tests
	Namespace string

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
	DockerCliConfig *dockershim.ClientConfig
	ContainerMgr    cm.ContainerManager     // Fake Container manager
	VPlugins        []volume.VolumePlugin   // Fake Volume Plugins
	Cli             *clientset.Clientset    // ClientSet
	HollowKubelet   *kubemark.HollowKubelet // Hallow kube mark
}

func createCli(t *testing.T) *clientset.Clientset {

	configFile := Cfg.RootDir + ConfigFile
	clientConfig, err := clientcmd.LoadFromFile(configFile)
	if err != nil {
		t.Fatalf("error while loading kubeconfig from file %v: %v", configFile, err)
	}

	config, err := clientcmd.NewDefaultClientConfig(*clientConfig, &clientcmd.ConfigOverrides{}).ClientConfig()
	if err != nil {
		t.Fatalf("error while creating kubeconfig: %v", err)
	}
	config.ContentType = "application/vnd.kubernetes.protobuf"
	config.QPS = 10
	config.Burst = 20

	Cli, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatalf("Failed to create a ClientSet: %v. Exiting.\n", err)
	}

	t.Logf("Created CLI\n")

	return Cli
}

//NewHollowNode creates a Hollow node structure
func NewHollowNode(t *testing.T, index int) *HollowNode {
	var hn *HollowNode
	Name := fmt.Sprintf("%s%d", NodeName, index)
	basePath := Cfg.RootDir
	nodePath := fmt.Sprintf("%s/nodes/%s", basePath, Name)
	createDIR(t, nodePath)
	hn = &HollowNode{Name: Name, NodePath: nodePath}

	hn.ConfigFile = basePath + ConfigFile

	hn.Cli = createCli(t)

	hn.CadvisorMgr = &cadvisortest.Fake{
		NodeName: Name,
	}
	hn.ContainerMgr = cm.NewStubContainerManager()

	hn.DockerCli = libdocker.NewFakeDockerClient().WithTraceDisabled()
	hn.DockerCli.EnableSleep = true

	hn.DockerCliConfig = &dockershim.ClientConfig{
		DockerEndpoint:    libdocker.FakeDockerEndpoint,
		EnableSleep:       true,
		WithTraceDisabled: true,
	}

	hollowKubelet := kubemark.NewHollowKubelet(
		Name,
		hn.Cli,
		hn.CadvisorMgr,
		hn.DockerCliConfig,
		10250+(index*10),
		10500+(index*10),
		hn.ContainerMgr,
		110,
		0,
	)

	hollowKubelet.KubeletFlags.RemoteRuntimeEndpoint = fmt.Sprintf("unix://%s/%s", nodePath, RemoteEndPoint)
	dockerShimDir := fmt.Sprintf("%s/dockershim", nodePath)
	createDIR(t, dockerShimDir)
	hollowKubelet.KubeletFlags.DockershimRootDirectory = dockerShimDir

	podManifest := fmt.Sprintf("%s/podManifest", nodePath)
	createDIR(t, podManifest)
	hollowKubelet.KubeletConfiguration.PodManifestPath = podManifest

	//Add an additional Fake volume plugin.
	_, fkvolumePlugin := fakevolume.GetTestVolumePluginMgr(t)
	hollowKubelet.KubeletDeps.VolumePlugins = append(hollowKubelet.KubeletDeps.VolumePlugins, fkvolumePlugin)

	hn.HollowKubelet = hollowKubelet

	return hn
}

func createDIR(t *testing.T, dirname string) {

	t.Logf("Creating directory %s\n", dirname)
	err := os.Mkdir(dirname, 0777)
	if err != nil {
		t.Fatalf("Unable to create the dir=%s err=%v", dirname, err)
	}
}

//Run simply call holow nodes Run() method in a go-routine
func (hn *HollowNode) Run() {

	go hn.HollowKubelet.Run()
	//Give it a second before starting another hollow kubelet
	time.Sleep(time.Second)
}

//HollowNodes Map of nodes we might need to lookup in the tests.
type HollowNodes struct {
	Nodes map[string]*HollowNode //A Map of hollow nodes

}

//Add simply adds the hollownode to the lookup map
func (hns *HollowNodes) Add(hn *HollowNode) {
	hns.Nodes[hn.Name] = hn
}

// ListContainers ListContainers prints all the containers from the fake Docker client from all the nodes
func (hns *HollowNodes) ListContainers() []string {

	var result []string
	var opt dockertypes.ContainerListOptions
	opt.All = true

	for _, n := range hns.Nodes {
		containers, _ := n.DockerCli.ListContainers(opt)
		for _, c := range containers {
			result = append(result, n.Name+":"+c.ID)
		}
	}

	return result
}

// InitNodes Initialize Initalize test suite to do the following
// 1) Create NumNodes * Hollow-Nodes
// 2) Monitor Kube-system processes such as ApiServer, Controller Manager and Scheduler
func InitNodes(t *testing.T, serverPath string, insecurePort int) error {

	// Get the root data
	Cfg.RootDir = serverPath
	Cfg.Master = fmt.Sprintf("http://localhost:%d", insecurePort)
	Cfg.Namespace = NameSpace
	Cfg.Nodes = &HollowNodes{Nodes: make(map[string]*HollowNode)}
	Cfg.Client = createCli(t)
	t.Logf("Creating %d Hollow-nodes...\n", NumNodes)

	createDIR(t, Cfg.RootDir+"/nodes")

	for i := 0; i < NumNodes; i++ {

		hn := NewHollowNode(t, i)
		Cfg.Nodes.Add(hn)
		hn.Run()
	}

	return nil

}
