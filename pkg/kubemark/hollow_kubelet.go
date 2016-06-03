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
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/volume/empty_dir"
	"k8s.io/kubernetes/test/utils"

	"github.com/golang/glog"
)

type HollowKubelet struct {
	KubeletConfig *kubeletapp.KubeletConfig
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
	testRootDir := utils.MakeTempDirOrDie("hollow-kubelet.", "")
	manifestFilePath := utils.MakeTempDirOrDie("manifest", testRootDir)
	glog.Infof("Using %s as root dir for hollow-kubelet", testRootDir)

	return &HollowKubelet{
		KubeletConfig: kubeletapp.SimpleKubelet(
			client,
			dockerClient,
			nodeName,
			testRootDir,
			"",        /* manifest-url */
			"0.0.0.0", /* bind address */
			uint(kubeletPort),
			uint(kubeletReadOnlyPort),
			api.NamespaceDefault,
			empty_dir.ProbeVolumePlugins(),
			nil, /* tls-options */
			cadvisorInterface,
			manifestFilePath,
			nil, /* cloud-provider */
			&containertest.FakeOS{}, /* os-interface */
			20*time.Second,          /* FileCheckFrequency */
			20*time.Second,          /* HTTPCheckFrequency */
			1*time.Minute,           /* MinimumGCAge */
			10*time.Second,          /* NodeStatusUpdateFrequency */
			10*time.Second,          /* SyncFrequency */
			5*time.Minute,           /* OutOfDiskTransitionFrequency */
			5*time.Minute,           /* EvictionPressureTransitionPeriod */
			maxPods,
			podsPerCore,
			containerManager,
			nil,
		),
	}
}

// Starts this HollowKubelet and blocks.
func (hk *HollowKubelet) Run() {
	kubeletapp.RunKubelet(hk.KubeletConfig)
	select {}
}
