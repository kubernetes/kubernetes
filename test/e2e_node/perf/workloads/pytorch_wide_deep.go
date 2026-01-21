/*
Copyright 2025 The Kubernetes Authors.

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

package workloads

import (
	"fmt"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

// pytorchWideDeepWorkload defines a workload to run a PyTorch Wide-Deep
// model training benchmark for CPU Manager validation.
type pytorchWideDeepWorkload struct{}

// Ensure pytorchWideDeepWorkload implements NodePerfWorkload interface.
var _ NodePerfWorkload = &pytorchWideDeepWorkload{}

func (w pytorchWideDeepWorkload) Name() string {
	return "pytorch-wide-deep"
}

func (w pytorchWideDeepWorkload) PodSpec() v1.PodSpec {
	var containers []v1.Container
	ctn := v1.Container{
		Name:  fmt.Sprintf("%s-ctn", w.Name()),
		Image: imageutils.GetE2EImage(imageutils.NodePerfPytorchWideDeep),
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU):    resource.MustParse("15000m"),
				v1.ResourceName(v1.ResourceMemory): resource.MustParse("16Gi"),
			},
			Limits: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU):    resource.MustParse("15000m"),
				v1.ResourceName(v1.ResourceMemory): resource.MustParse("16Gi"),
			},
		},
		// The container entrypoint already runs: time -p python /train_wide_deep.py
	}
	containers = append(containers, ctn)

	return v1.PodSpec{
		RestartPolicy: v1.RestartPolicyNever,
		Containers:    containers,
	}
}

func (w pytorchWideDeepWorkload) Timeout() time.Duration {
	return 15 * time.Minute
}

func (w pytorchWideDeepWorkload) KubeletConfig(oldCfg *kubeletconfig.KubeletConfiguration) (newCfg *kubeletconfig.KubeletConfiguration, err error) {
	// Enable CPU Manager in Kubelet with static policy.
	newCfg = oldCfg.DeepCopy()
	// Set the CPU Manager policy to static.
	newCfg.CPUManagerPolicy = string(cpumanager.PolicyStatic)
	// Set the CPU Manager reconcile period to 10 second.
	newCfg.CPUManagerReconcilePeriod = metav1.Duration{Duration: 10 * time.Second}

	// The Kubelet panics if either kube-reserved or system-reserved is not set
	// when static CPU Manager is enabled. Set cpu in kube-reserved > 0 so that
	// kubelet doesn't panic.
	if newCfg.KubeReserved == nil {
		newCfg.KubeReserved = map[string]string{}
	}

	if _, ok := newCfg.KubeReserved["cpu"]; !ok {
		newCfg.KubeReserved["cpu"] = "200m"
	}

	return newCfg, nil
}

func (w pytorchWideDeepWorkload) PreTestExec() error {
	cmd := "/bin/sh"
	args := []string{"-c", "rm -f /var/lib/kubelet/cpu_manager_state"}
	err := runCmd(cmd, args)

	return err
}

func (w pytorchWideDeepWorkload) PostTestExec() error {
	cmd := "/bin/sh"
	args := []string{"-c", "rm -f /var/lib/kubelet/cpu_manager_state"}
	err := runCmd(cmd, args)

	return err
}

func (w pytorchWideDeepWorkload) ExtractPerformanceFromLogs(logs string) (perf time.Duration, err error) {
	perfLine, err := getMatchingLineFromLog(logs, "real")
	if err != nil {
		return perf, err
	}
	perfString := fmt.Sprintf("%ss", strings.TrimSpace(strings.TrimPrefix(perfLine, "real")))
	perf, err = time.ParseDuration(perfString)

	return perf, err
}
