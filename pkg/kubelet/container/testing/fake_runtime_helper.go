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

package testing

import (
	"k8s.io/api/core/v1"
	kubetypes "k8s.io/apimachinery/pkg/types"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// FakeRuntimeHelper implements RuntimeHelper interfaces for testing purposes.
type FakeRuntimeHelper struct {
	DNSServers      []string
	DNSSearches     []string
	DNSOptions      []string
	HostName        string
	HostDomain      string
	PodContainerDir string
	Err             error
}

// GenerateRunContainerOptions on fake runtime helper generates fake RunContainerOptions.
// It is not used by docker runtime.
func (f *FakeRuntimeHelper) GenerateRunContainerOptions(pod *v1.Pod, container *v1.Container, podIP string, podIPs []string) (*kubecontainer.RunContainerOptions, func(), error) {
	var opts kubecontainer.RunContainerOptions
	if len(container.TerminationMessagePath) != 0 {
		opts.PodContainerDir = f.PodContainerDir
	}
	return &opts, nil, nil
}

// GetPodCgroupParent on fake runtime helper returns fake pod cgroup parent.
// It is not used by docker runtime.
func (f *FakeRuntimeHelper) GetPodCgroupParent(pod *v1.Pod) string {
	return ""
}

// GetPodDNS on fake runtime helper returns fake DNS settings for pod.
// It is not used by docker runtime.
func (f *FakeRuntimeHelper) GetPodDNS(pod *v1.Pod) (*runtimeapi.DNSConfig, error) {
	return &runtimeapi.DNSConfig{
		Servers:  f.DNSServers,
		Searches: f.DNSSearches,
		Options:  f.DNSOptions}, f.Err
}

// GeneratePodHostNameAndDomain on fake runtime helper returns fake hostname and domain.
// It is not used by docker runtime.
func (f *FakeRuntimeHelper) GeneratePodHostNameAndDomain(pod *v1.Pod) (string, string, error) {
	return f.HostName, f.HostDomain, f.Err
}

// GetPodDir on fake runtime helper returns fake path to the per-pod data directory for the
// specified pod. It is not used by docker runtime.
func (f *FakeRuntimeHelper) GetPodDir(podUID kubetypes.UID) string {
	return "/poddir/" + string(podUID)
}

// GetExtraSupplementalGroupsForPod on fake runtime helper returns a
// fake list of extra supplemental groups for pod.
// It is not used by docker runtime.
func (f *FakeRuntimeHelper) GetExtraSupplementalGroupsForPod(pod *v1.Pod) []int64 {
	return nil
}
