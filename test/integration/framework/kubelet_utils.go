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
	"fmt"
	"io/ioutil"
	"net"
	"os"

	kubeletapp "github.com/GoogleCloudPlatform/kubernetes/cmd/kubelet/app"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/cadvisor"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	nodeutil "github.com/GoogleCloudPlatform/kubernetes/pkg/util/node"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume/empty_dir"

	"github.com/golang/glog"
)

// StartPodWorkers starts pod workers for the given pods.
func StartPodWorkers(k *kubelet.Kubelet, pods *api.PodList, wait bool) {
	// Start pod workers in parallel
	RunParallel(func(id int) error {
		k.NotifyPodWorker(&pods.Items[id], nil, func() {})
		return nil
	}, len(pods.Items), 0)

	if !wait {
		return
	}
	podUpdates := k.GetPodStatusChannel()
	for i := 0; i < len(pods.Items); i++ {
		update := <-podUpdates
		if update.Status.Phase != api.PodRunning {
			glog.Fatalf("Pod %v not in Running, status: %+v", update.Pod.Name, update.Status)
		}
	}

}

// GetPods returns a list of pods. Note the type.
func GetPods(k *kubelet.Kubelet, all bool) []*container.Pod {
	runningPods, err := k.GetContainerRuntime().GetPods(all)
	if err != nil {
		glog.Fatalf("Could not get pods: %v", err)
	}
	return runningPods
}

// NewPodList creates a list of pods assigned to the given host with the given status.
func NewPodList(count int, hostname string, status api.PodPhase) *api.PodList {
	pods := []api.Pod{}
	for i := 0; i < count; i++ {
		newPod := api.Pod{
			TypeMeta: api.TypeMeta{
				Kind:       "Pod",
				APIVersion: testapi.Version(),
			},
			ObjectMeta: api.ObjectMeta{
				Name: fmt.Sprintf("pod%d", i),
				UID:  util.NewUUID(),
				// TODO: Pass in the labels
				Labels:    map[string]string{"name": "foo"},
				Namespace: TestNS,
			},
			Status: api.PodStatus{
				Phase: status,
			},
			Spec: api.PodSpec{
				NodeName: hostname,
				Containers: []api.Container{
					{
						Name:  "bar",
						Image: "kubernetes/pause",
					},
				},
			},
		}
		pods = append(pods, newPod)
	}
	return &api.PodList{
		Items: pods,
	}
}

// TestKubeletConfig is used to configure the test kubelet.
type TestKubeletConfig struct {
	// The restclient the kubelet uses to talk to the apiserver. For normal kubelet
	// benchmarks this shouldn't be necessary. If you need a master, you can call
	// NewMasterComponents just like the master benchmarks.
	// TODO: Pipe the restclient to the kubelet gracefully, if you need this
	// behavior today you need to modify the call to NewMainKubelet below.
	RestClient *client.Client
	// Use a fake docker if true.
	FakeDocker bool
	// Start the status manager if true. Note that this option only makes sense with
	// a running apiserver and working restclient.
	StartStatusManager bool
	// Start watchers for services and nodes. Note that this option only makes sense
	// with a running apiserver and working restclient.
	StartWatchers bool
}

// CreateKubeletOrDie creates a kubelet based on the given TestKubeletConfig.
func CreateKubeletOrDie(tkc *TestKubeletConfig) *kubelet.Kubelet {
	dockerEndpoint := ""
	if tkc.FakeDocker {
		dockerEndpoint = "fake://"
	}
	dockerClient := dockertools.ConnectToDockerOrDie(dockerEndpoint)
	testRootDir := makeTempDirOrDie("kubelet_integ_1.", "")
	configFilePath := makeTempDirOrDie("config", testRootDir)
	cadvisorInterface := new(cadvisor.Fake)

	kc := kubeletapp.SimpleKubelet(
		tkc.RestClient, dockerClient, "localhost", testRootDir, "", "127.0.0.1", 10250,
		api.NamespaceDefault, empty_dir.ProbeVolumePlugins(), nil,
		cadvisorInterface, configFilePath, nil, kubecontainer.FakeOS{})

	// TODO: Enable/Disable via test config
	//pc = makePodSourceConfig(kc)

	gcPolicy := kubelet.ContainerGCPolicy{
		MinAge:             kc.MinimumGCAge,
		MaxPerPodContainer: kc.MaxPerPodContainerCount,
		MaxContainers:      kc.MaxContainerCount,
	}

	kStopCh := make(chan struct{})
	k, err := kubelet.NewMainKubelet(
		nodeutil.GetHostname(kc.HostnameOverride),
		kc.DockerClient,
		// TODO: Pipe this through from caller when we need a client. This is hardcoded as nil
		// because the type is an interface.
		nil,
		kc.RootDirectory,
		kc.PodInfraContainerImage,
		kc.SyncFrequency,
		float32(kc.RegistryPullQPS),
		kc.RegistryBurst,
		gcPolicy,
		// SeenAllSources,
		func() bool { return true },
		kc.RegisterNode,
		kc.StandaloneMode,
		kc.ClusterDomain,
		net.IP(kc.ClusterDNS),
		kc.MasterServiceNamespace,
		kc.VolumePlugins,
		kc.NetworkPlugins,
		kc.NetworkPluginName,
		kc.StreamingConnectionIdleTimeout,
		&record.FakeRecorder{},
		kc.CadvisorInterface,
		kc.ImageGCPolicy,
		kc.DiskSpacePolicy,
		kc.Cloud,
		kc.NodeStatusUpdateFrequency,
		kc.ResourceContainer,
		kc.OSInterface,
		kc.CgroupRoot,
		kc.ContainerRuntime,
		kc.Mounter,
		kc.DockerDaemonContainer,
		kc.SystemContainer,
		kc.ConfigureCBR0,
		kc.MaxPods,
		kc.DockerExecHandler,
		kStopCh,
	)

	if err != nil {
		glog.Fatalf("Unexpected error %v", err)
	}
	if !tkc.StartWatchers {
		// We are going to hand populate the kubelet stores, so turn off all watchers.
		close(kStopCh)
	}

	node := &api.Node{
		ObjectMeta: api.ObjectMeta{Name: "localhost"},
		Status: api.NodeStatus{
			Addresses: []api.NodeAddress{
				{
					Type:    api.NodeLegacyHostIP,
					Address: "127.0.0.1",
				},
			},
		},
	}
	if nl, ok := k.NodeLister.(*cache.StoreToNodeLister); ok {
		nl.Store.Replace([]interface{}{node})
	} else {
		glog.Fatalf("Unable to insert node into kubelet's nodestore")
	}

	if _, err := k.GetNode(); err != nil {
		glog.Fatalf("Unexpected error %v", err)
	}
	if tkc.StartStatusManager {
		k.StartStatusManager()
	}
	return k
}

func makeTempDirOrDie(prefix string, baseDir string) string {
	if baseDir == "" {
		baseDir = "/tmp"
	}
	tempDir, err := ioutil.TempDir(baseDir, prefix)
	if err != nil {
		glog.Fatalf("Can't make a temp rootdir: %v", err)
	}
	if err = os.MkdirAll(tempDir, 0750); err != nil {
		glog.Fatalf("Can't mkdir(%q): %v", tempDir, err)
	}
	return tempDir
}
