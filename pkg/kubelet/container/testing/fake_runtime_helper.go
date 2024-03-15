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
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	kubetypes "k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/kubernetes/pkg/features"
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
	RuntimeHandlers map[string]kubecontainer.RuntimeHandler
	Err             error
}

func (f *FakeRuntimeHelper) GenerateRunContainerOptions(_ context.Context, pod *v1.Pod, container *v1.Container, podIP string, podIPs []string) (*kubecontainer.RunContainerOptions, func(), error) {
	var opts kubecontainer.RunContainerOptions
	if len(container.TerminationMessagePath) != 0 {
		opts.PodContainerDir = f.PodContainerDir
	}
	return &opts, nil, nil
}

func (f *FakeRuntimeHelper) GetPodCgroupParent(pod *v1.Pod) string {
	return ""
}

func (f *FakeRuntimeHelper) GetPodDNS(pod *v1.Pod) (*runtimeapi.DNSConfig, error) {
	return &runtimeapi.DNSConfig{
		Servers:  f.DNSServers,
		Searches: f.DNSSearches,
		Options:  f.DNSOptions}, f.Err
}

// This is not used by docker runtime.
func (f *FakeRuntimeHelper) GeneratePodHostNameAndDomain(pod *v1.Pod) (string, string, error) {
	return f.HostName, f.HostDomain, f.Err
}

func (f *FakeRuntimeHelper) GetPodDir(podUID kubetypes.UID) string {
	return "/poddir/" + string(podUID)
}

func (f *FakeRuntimeHelper) GetExtraSupplementalGroupsForPod(pod *v1.Pod) []int64 {
	return nil
}

func (f *FakeRuntimeHelper) GetOrCreateUserNamespaceMappings(pod *v1.Pod, runtimeHandler string) (*runtimeapi.UserNamespace, error) {
	featureEnabled := utilfeature.DefaultFeatureGate.Enabled(features.UserNamespacesSupport)
	if pod == nil || pod.Spec.HostUsers == nil {
		return nil, nil
	}
	// pod.Spec.HostUsers is set to true/false
	if !featureEnabled {
		return nil, fmt.Errorf("the feature gate %q is disabled: can't set spec.HostUsers", features.UserNamespacesSupport)
	}
	if *pod.Spec.HostUsers {
		return nil, nil
	}

	// From here onwards, hostUsers=false and the feature gate is enabled.

	// if the pod requested a user namespace and the runtime doesn't support user namespaces then return an error.
	if h, ok := f.RuntimeHandlers[runtimeHandler]; !ok {
		return nil, fmt.Errorf("RuntimeClass handler %q not found", runtimeHandler)
	} else if !h.SupportsUserNamespaces {
		return nil, fmt.Errorf("RuntimeClass handler %q does not support user namespaces", runtimeHandler)
	}

	ids := &runtimeapi.IDMapping{
		HostId:      65536,
		ContainerId: 0,
		Length:      65536,
	}

	return &runtimeapi.UserNamespace{
		Mode: runtimeapi.NamespaceMode_POD,
		Uids: []*runtimeapi.IDMapping{ids},
		Gids: []*runtimeapi.IDMapping{ids},
	}, nil
}

func (f *FakeRuntimeHelper) PrepareDynamicResources(pod *v1.Pod) error {
	return nil
}

func (f *FakeRuntimeHelper) UnprepareDynamicResources(pod *v1.Pod) error {
	return nil
}
