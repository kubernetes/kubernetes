/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package dockertools

import (
	cadvisorApi "github.com/google/cadvisor/info/v1"
	"k8s.io/kubernetes/pkg/client/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/prober"
	kubeletTypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/util/oom"
	"k8s.io/kubernetes/pkg/util/procfs"
)

func NewFakeDockerManager(
	client DockerInterface,
	recorder record.EventRecorder,
	readinessManager *kubecontainer.ReadinessManager,
	containerRefManager *kubecontainer.RefManager,
	machineInfo *cadvisorApi.MachineInfo,
	podInfraContainerImage string,
	qps float32,
	burst int,
	containerLogsDir string,
	osInterface kubecontainer.OSInterface,
	networkPlugin network.NetworkPlugin,
	generator kubecontainer.RunContainerOptionsGenerator,
	httpClient kubeletTypes.HttpGetter) *DockerManager {

	fakeOomAdjuster := oom.NewFakeOomAdjuster()
	fakeProcFs := procfs.NewFakeProcFs()
	dm := NewDockerManager(client, recorder, readinessManager, containerRefManager, machineInfo, podInfraContainerImage, qps,
		burst, containerLogsDir, osInterface, networkPlugin, generator, httpClient, &NativeExecHandler{},
		fakeOomAdjuster, fakeProcFs, false)
	dm.dockerPuller = &FakeDockerPuller{}
	dm.prober = prober.New(nil, readinessManager, containerRefManager, recorder)
	return dm
}
