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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	kubeletapp "k8s.io/kubernetes/cmd/kubelet/app"
	"k8s.io/kubernetes/cmd/kubelet/app/options"
	"k8s.io/kubernetes/pkg/kubelet"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/dockershim"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/oom"
	"k8s.io/kubernetes/pkg/volume/emptydir"
	"k8s.io/kubernetes/pkg/volume/projected"
	"k8s.io/kubernetes/pkg/volume/secret"
	"k8s.io/kubernetes/pkg/volume/util/subpath"
	"k8s.io/kubernetes/test/utils"

	"k8s.io/klog"
)

type HollowKubelet struct {
	KubeletFlags         *options.KubeletFlags
	KubeletConfiguration *kubeletconfig.KubeletConfiguration
	KubeletDeps          *kubelet.Dependencies
}

func NewHollowKubelet(
	nodeName string,
	client *clientset.Clientset,
	cadvisorInterface cadvisor.Interface,
	dockerClientConfig *dockershim.ClientConfig,
	kubeletPort, kubeletReadOnlyPort int,
	containerManager cm.ContainerManager,
	maxPods int, podsPerCore int,
) *HollowKubelet {
	// -----------------
	// Static config
	// -----------------
	f, c := GetHollowKubeletConfig(nodeName, kubeletPort, kubeletReadOnlyPort, maxPods, podsPerCore)

	// -----------------
	// Injected objects
	// -----------------
	volumePlugins := emptydir.ProbeVolumePlugins()
	volumePlugins = append(volumePlugins, secret.ProbeVolumePlugins()...)
	volumePlugins = append(volumePlugins, projected.ProbeVolumePlugins()...)
	d := &kubelet.Dependencies{
		KubeClient:         client,
		HeartbeatClient:    client,
		DockerClientConfig: dockerClientConfig,
		CAdvisorInterface:  cadvisorInterface,
		Cloud:              nil,
		OSInterface:        &containertest.FakeOS{},
		ContainerManager:   containerManager,
		VolumePlugins:      volumePlugins,
		TLSOptions:         nil,
		OOMAdjuster:        oom.NewFakeOOMAdjuster(),
		Mounter:            mount.New("" /* default mount path */),
		Subpather:          &subpath.FakeSubpath{},
	}

	return &HollowKubelet{
		KubeletFlags:         f,
		KubeletConfiguration: c,
		KubeletDeps:          d,
	}
}

// Starts this HollowKubelet and blocks.
func (hk *HollowKubelet) Run() {
	if err := kubeletapp.RunKubelet(&options.KubeletServer{
		KubeletFlags:         *hk.KubeletFlags,
		KubeletConfiguration: *hk.KubeletConfiguration,
	}, hk.KubeletDeps, false); err != nil {
		klog.Fatalf("Failed to run HollowKubelet: %v. Exiting.", err)
	}
	select {}
}

// Builds a KubeletConfiguration for the HollowKubelet, ensuring that the
// usual defaults are applied for fields we do not override.
func GetHollowKubeletConfig(
	nodeName string,
	kubeletPort int,
	kubeletReadOnlyPort int,
	maxPods int,
	podsPerCore int) (*options.KubeletFlags, *kubeletconfig.KubeletConfiguration) {

	testRootDir := utils.MakeTempDirOrDie("hollow-kubelet.", "")
	podFilePath := utils.MakeTempDirOrDie("static-pods", testRootDir)
	klog.Infof("Using %s as root dir for hollow-kubelet", testRootDir)

	// Flags struct
	f := options.NewKubeletFlags()
	f.EnableServer = true
	f.RootDirectory = testRootDir
	f.HostnameOverride = nodeName
	f.MinimumGCAge = metav1.Duration{Duration: 1 * time.Minute}
	f.MaxContainerCount = 100
	f.MaxPerPodContainerCount = 2
	f.RegisterNode = true
	f.RegisterSchedulable = true

	// Config struct
	c, err := options.NewKubeletConfiguration()
	if err != nil {
		panic(err)
	}

	c.StaticPodURL = ""
	c.Address = "0.0.0.0" /* bind address */
	c.Port = int32(kubeletPort)
	c.ReadOnlyPort = int32(kubeletReadOnlyPort)
	c.StaticPodPath = podFilePath
	c.FileCheckFrequency.Duration = 20 * time.Second
	c.HTTPCheckFrequency.Duration = 20 * time.Second
	c.NodeStatusUpdateFrequency.Duration = 10 * time.Second
	c.SyncFrequency.Duration = 10 * time.Second
	c.EvictionPressureTransitionPeriod.Duration = 5 * time.Minute
	c.MaxPods = int32(maxPods)
	c.PodsPerCore = int32(podsPerCore)
	c.ClusterDNS = []string{}
	c.ImageGCHighThresholdPercent = 90
	c.ImageGCLowThresholdPercent = 80
	c.VolumeStatsAggPeriod.Duration = time.Minute
	c.CgroupRoot = ""
	c.CPUCFSQuota = true
	c.EnableControllerAttachDetach = false
	c.EnableDebuggingHandlers = true
	c.CgroupsPerQOS = false
	// hairpin-veth is used to allow hairpin packets. Note that this deviates from
	// what the "real" kubelet currently does, because there's no way to
	// set promiscuous mode on docker0.
	c.HairpinMode = kubeletconfig.HairpinVeth
	c.MaxOpenFiles = 1024
	c.RegistryBurst = 10
	c.RegistryPullQPS = 5.0
	c.ResolverConfig = kubetypes.ResolvConfDefault
	c.KubeletCgroups = "/kubelet"
	c.SerializeImagePulls = true
	c.SystemCgroups = ""
	c.ProtectKernelDefaults = false

	return f, c
}
