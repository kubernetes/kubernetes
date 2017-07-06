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

package selfhosting

import (
	"k8s.io/api/core/v1"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

// mutatePodSpec makes a Static Pod-hosted PodSpec suitable for self-hosting
func mutatePodSpec(name string, podSpec *v1.PodSpec) {
	mutators := map[string][]func(*v1.PodSpec){
		kubeAPIServer: {
			addNodeSelectorToPodSpec,
			setMasterTolerationOnPodSpec,
			setRightDNSPolicyOnPodSpec,
		},
		kubeControllerManager: {
			addNodeSelectorToPodSpec,
			setMasterTolerationOnPodSpec,
			setRightDNSPolicyOnPodSpec,
		},
		kubeScheduler: {
			addNodeSelectorToPodSpec,
			setMasterTolerationOnPodSpec,
			setRightDNSPolicyOnPodSpec,
		},
	}

	// Get the mutator functions for the component in question, then loop through and execute them
	mutatorsForComponent := mutators[name]
	for _, mutateFunc := range mutatorsForComponent {
		mutateFunc(podSpec)
	}
}

// addNodeSelectorToPodSpec makes Pod require to be scheduled on a node marked with the master label
func addNodeSelectorToPodSpec(podSpec *v1.PodSpec) {
	if podSpec.NodeSelector == nil {
		podSpec.NodeSelector = map[string]string{kubeadmconstants.LabelNodeRoleMaster: ""}
		return
	}

	podSpec.NodeSelector[kubeadmconstants.LabelNodeRoleMaster] = ""
}

// setMasterTolerationOnPodSpec makes the Pod tolerate the master taint
func setMasterTolerationOnPodSpec(podSpec *v1.PodSpec) {
	if podSpec.Tolerations == nil {
		podSpec.Tolerations = []v1.Toleration{kubeadmconstants.MasterToleration}
		return
	}

	podSpec.Tolerations = append(podSpec.Tolerations, kubeadmconstants.MasterToleration)
}

// setRightDNSPolicyOnPodSpec makes sure the self-hosted components can look up things via kube-dns if necessary
func setRightDNSPolicyOnPodSpec(podSpec *v1.PodSpec) {
	podSpec.DNSPolicy = v1.DNSClusterFirstWithHostNet
}
