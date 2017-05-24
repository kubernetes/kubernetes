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

package kubemark

import (
	"time"

	kubeletapp "k8s.io/kubernetes/cmd/kubelet/app"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/kubelet"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	kubeio "k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/oom"
	"k8s.io/kubernetes/pkg/volume/empty_dir"
	"k8s.io/kubernetes/pkg/volume/secret"
	"k8s.io/kubernetes/test/utils"

	"github.com/golang/glog"
)

type HollowKubelet struct {
	KubeletConfiguration *componentconfig.KubeletConfiguration
	KubeletDeps          *kubelet.KubeletDeps
}

func NewHollowKubelet(
	nodeName string,
	client *clientset.Clientset,
	cadvisorInterface cadvisor.Interface,
	dockerClient dockertools.DockerInterface,
	kubeletPort, kubeletReadOnlyPort int,
	containerManager cm.ContainerManager,
	maxPods int, podsPerCore int,
) *HollowKubelet {
	// -----------------
	// Static config
	// -----------------
	c := GetHollowKubeletConfig(nodeName, kubeletPort, kubeletReadOnlyPort, maxPods, podsPerCore)

	// -----------------
	// Injected objects
	// -----------------
	volumePlugins := empty_dir.ProbeVolumePlugins()
	volumePlugins = append(volumePlugins, secret.ProbeVolumePlugins()...)
	d := &kubelet.KubeletDeps{
		KubeClient:        client,
		DockerClient:      dockerClient,
		CAdvisorInterface: cadvisorInterface,
		Cloud:             nil,
		OSInterface:       &containertest.FakeOS{},
		ContainerManager:  containerManager,
		VolumePlugins:     volumePlugins,
		TLSOptions:        nil,
		OOMAdjuster:       oom.NewFakeOOMAdjuster(),
		Writer:            &kubeio.StdWriter{},
		Mounter:           mount.New("" /* default mount path */),
	}

	return &HollowKubelet{
		KubeletConfiguration: c,
		KubeletDeps:          d,
	}
}

// Starts this HollowKubelet and blocks.
func (hk *HollowKubelet) Run() {
	kubeletapp.RunKubelet(hk.KubeletConfiguration, hk.KubeletDeps, false, false)
	select {}
}

// Builds a KubeletConfiguration for the HollowKubelet, ensuring that the
// usual defaults are applied for fields we do not override.
func GetHollowKubeletConfig(
	nodeName string,
	kubeletPort int,
	kubeletReadOnlyPort int,
	maxPods int,
	podsPerCore int) *componentconfig.KubeletConfiguration {

	testRootDir := utils.MakeTempDirOrDie("hollow-kubelet.", "")
	manifestFilePath := utils.MakeTempDirOrDie("manifest", testRootDir)
	glog.Infof("Using %s as root dir for hollow-kubelet", testRootDir)

	// Do the external -> internal conversion to make sure that defaults
	// are set for fields not overridden in NewHollowKubelet.
	tmp := &v1alpha1.KubeletConfiguration{}
	api.Scheme.Default(tmp)
	c := &componentconfig.KubeletConfiguration{}
	api.Scheme.Convert(tmp, c, nil)

	c.HostnameOverride = nodeName
	c.RootDirectory = testRootDir
	c.ManifestURL = ""
	c.Address = "0.0.0.0" /* bind address */
	c.Port = int32(kubeletPort)
	c.ReadOnlyPort = int32(kubeletReadOnlyPort)
	c.MasterServiceNamespace = api.NamespaceDefault
	c.PodManifestPath = manifestFilePath
	c.FileCheckFrequency.Duration = 20 * time.Second
	c.HTTPCheckFrequency.Duration = 20 * time.Second
	c.MinimumGCAge.Duration = 1 * time.Minute
	c.NodeStatusUpdateFrequency.Duration = 10 * time.Second
	c.SyncFrequency.Duration = 10 * time.Second
	c.OutOfDiskTransitionFrequency.Duration = 5 * time.Minute
	c.EvictionPressureTransitionPeriod.Duration = 5 * time.Minute
	c.MaxPods = int32(maxPods)
	c.PodsPerCore = int32(podsPerCore)
	c.ClusterDNS = ""
	c.DockerExecHandlerName = "native"
	c.ImageGCHighThresholdPercent = 90
	c.ImageGCLowThresholdPercent = 80
	c.LowDiskSpaceThresholdMB = 256
	c.VolumeStatsAggPeriod.Duration = time.Minute
	c.CgroupRoot = ""
	c.ContainerRuntime = "docker"
	c.CPUCFSQuota = true
	c.RuntimeCgroups = ""
	c.EnableControllerAttachDetach = false
	c.EnableCustomMetrics = false
	c.EnableDebuggingHandlers = true
	c.EnableServer = true
	c.ExperimentalCgroupsPerQOS = false
	// hairpin-veth is used to allow hairpin packets. Note that this deviates from
	// what the "real" kubelet currently does, because there's no way to
	// set promiscuous mode on docker0.
	c.HairpinMode = componentconfig.HairpinVeth
	c.MaxContainerCount = 100
	c.MaxOpenFiles = 1024
	c.MaxPerPodContainerCount = 2
	c.NvidiaGPUs = 0
	c.RegisterNode = true
	c.RegisterSchedulable = true
	c.RegistryBurst = 10
	c.RegistryPullQPS = 5.0
	c.ResolverConfig = kubetypes.ResolvConfDefault
	c.KubeletCgroups = "/kubelet"
	c.SerializeImagePulls = true
	c.SystemCgroups = ""
	c.ProtectKernelDefaults = false

	// TODO(mtaufen): Note that PodInfraContainerImage was being set to the empty value before,
	//                but this may not have been intentional. (previous code (SimpleKubelet)
	//                was peeling it off of a componentconfig.KubeletConfiguration{}, but may
	//                have actually wanted the default).
	//                The default will be present in the KubeletConfiguration contstructed above.

	return c

}
