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

package workloads

import (
	"fmt"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
)

// npbISWorkload defines a workload to run the integer sort (IS) workload
// from NAS parallel benchmark (NPB) suite.
type npbISWorkload struct{}

// Ensure npbISWorkload implemets NodePerfWorkload interface.
var _ NodePerfWorkload = &npbISWorkload{}

func (w npbISWorkload) Name() string {
	return "npb-is"
}

func (w npbISWorkload) PodSpec() corev1.PodSpec {
	var containers []corev1.Container
	ctn := corev1.Container{
		Name:  fmt.Sprintf("%s-ctn", w.Name()),
		Image: "gcr.io/kubernetes-e2e-test-images/node-perf/npb-is:1.0",
		Resources: corev1.ResourceRequirements{
			Requests: corev1.ResourceList{
				corev1.ResourceName(corev1.ResourceCPU):    resource.MustParse("16000m"),
				corev1.ResourceName(corev1.ResourceMemory): resource.MustParse("48Gi"),
			},
			Limits: corev1.ResourceList{
				corev1.ResourceName(corev1.ResourceCPU):    resource.MustParse("16000m"),
				corev1.ResourceName(corev1.ResourceMemory): resource.MustParse("48Gi"),
			},
		},
		Command: []string{"/bin/sh"},
		Args:    []string{"-c", "/is.D.x"},
	}
	containers = append(containers, ctn)

	return corev1.PodSpec{
		RestartPolicy: corev1.RestartPolicyNever,
		Containers:    containers,
	}
}

func (w npbISWorkload) Timeout() time.Duration {
	return 4 * time.Minute
}

func (w npbISWorkload) KubeletConfig(oldCfg *kubeletconfig.KubeletConfiguration) (newCfg *kubeletconfig.KubeletConfiguration, err error) {
	return oldCfg, nil
}

func (w npbISWorkload) PreTestExec() error {
	return nil
}

func (w npbISWorkload) PostTestExec() error {
	return nil
}

func (w npbISWorkload) ExtractPerformanceFromLogs(logs string) (perf time.Duration, err error) {
	perfLine, err := getMatchingLineFromLog(logs, "Time in seconds =")
	if err != nil {
		return perf, err
	}
	perfStrings := strings.Split(perfLine, "=")
	perfString := fmt.Sprintf("%ss", strings.TrimSpace(perfStrings[1]))
	perf, err = time.ParseDuration(perfString)

	return perf, err
}
