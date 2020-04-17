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
	"path/filepath"
	"strings"

	v1 "k8s.io/api/core/v1"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

const (
	// selfHostedKubeConfigDir sets the directory where kubeconfig files for the scheduler and controller-manager should be mounted
	// Due to how the projected volume mount works (can only be a full directory, not mount individual files), we must change this from
	// the default as mounts cannot be nested (/etc/kubernetes would override /etc/kubernetes/pki)
	selfHostedKubeConfigDir = "/etc/kubernetes/kubeconfig"
)

// PodSpecMutatorFunc is a function capable of mutating a PodSpec
type PodSpecMutatorFunc func(*v1.PodSpec)

// GetDefaultMutators gets the mutator functions that always should be used
func GetDefaultMutators() map[string][]PodSpecMutatorFunc {
	return map[string][]PodSpecMutatorFunc{
		kubeadmconstants.KubeAPIServer: {
			addNodeSelectorToPodSpec,
			setControlPlaneTolerationOnPodSpec,
			setRightDNSPolicyOnPodSpec,
			setHostIPOnPodSpec,
		},
		kubeadmconstants.KubeControllerManager: {
			addNodeSelectorToPodSpec,
			setControlPlaneTolerationOnPodSpec,
			setRightDNSPolicyOnPodSpec,
		},
		kubeadmconstants.KubeScheduler: {
			addNodeSelectorToPodSpec,
			setControlPlaneTolerationOnPodSpec,
			setRightDNSPolicyOnPodSpec,
		},
	}
}

// GetMutatorsFromFeatureGates returns all mutators needed based on the feature gates passed
func GetMutatorsFromFeatureGates(certsInSecrets bool) map[string][]PodSpecMutatorFunc {
	// Here the map of different mutators to use for the control plane's podspec is stored
	mutators := GetDefaultMutators()

	if certsInSecrets {
		// Some extra work to be done if we should store the control plane certificates in Secrets
		// Add the store-certs-in-secrets-specific mutators here so that the self-hosted component starts using them
		mutators[kubeadmconstants.KubeAPIServer] = append(mutators[kubeadmconstants.KubeAPIServer], setSelfHostedVolumesForAPIServer)
		mutators[kubeadmconstants.KubeControllerManager] = append(mutators[kubeadmconstants.KubeControllerManager], setSelfHostedVolumesForControllerManager)
		mutators[kubeadmconstants.KubeScheduler] = append(mutators[kubeadmconstants.KubeScheduler], setSelfHostedVolumesForScheduler)
	}

	return mutators
}

// mutatePodSpec makes a Static Pod-hosted PodSpec suitable for self-hosting
func mutatePodSpec(mutators map[string][]PodSpecMutatorFunc, name string, podSpec *v1.PodSpec) {
	// Get the mutator functions for the component in question, then loop through and execute them
	mutatorsForComponent := mutators[name]
	for _, mutateFunc := range mutatorsForComponent {
		mutateFunc(podSpec)
	}
}

// addNodeSelectorToPodSpec makes Pod require to be scheduled on a node marked with the control-plane label
func addNodeSelectorToPodSpec(podSpec *v1.PodSpec) {
	if podSpec.NodeSelector == nil {
		podSpec.NodeSelector = map[string]string{kubeadmconstants.LabelNodeRoleMaster: ""}
		return
	}

	podSpec.NodeSelector[kubeadmconstants.LabelNodeRoleMaster] = ""
}

// setControlPlaneTolerationOnPodSpec makes the Pod tolerate the control-plane taint
func setControlPlaneTolerationOnPodSpec(podSpec *v1.PodSpec) {
	if podSpec.Tolerations == nil {
		podSpec.Tolerations = []v1.Toleration{kubeadmconstants.ControlPlaneToleration}
		return
	}

	podSpec.Tolerations = append(podSpec.Tolerations, kubeadmconstants.ControlPlaneToleration)
}

// setHostIPOnPodSpec sets the environment variable HOST_IP using downward API
func setHostIPOnPodSpec(podSpec *v1.PodSpec) {
	envVar := v1.EnvVar{
		Name: "HOST_IP",
		ValueFrom: &v1.EnvVarSource{
			FieldRef: &v1.ObjectFieldSelector{
				FieldPath: "status.hostIP",
			},
		},
	}

	podSpec.Containers[0].Env = append(podSpec.Containers[0].Env, envVar)

	for i := range podSpec.Containers[0].Command {
		if strings.Contains(podSpec.Containers[0].Command[i], "advertise-address") {
			podSpec.Containers[0].Command[i] = "--advertise-address=$(HOST_IP)"
		}
	}
}

// setRightDNSPolicyOnPodSpec makes sure the self-hosted components can look up things via kube-dns if necessary
func setRightDNSPolicyOnPodSpec(podSpec *v1.PodSpec) {
	podSpec.DNSPolicy = v1.DNSClusterFirstWithHostNet
}

// setSelfHostedVolumesForAPIServer makes sure the self-hosted api server has the right volume source coming from a self-hosted cluster
func setSelfHostedVolumesForAPIServer(podSpec *v1.PodSpec) {
	for i, v := range podSpec.Volumes {
		// If the volume name matches the expected one; switch the volume source from hostPath to cluster-hosted
		if v.Name == kubeadmconstants.KubeCertificatesVolumeName {
			podSpec.Volumes[i].VolumeSource = apiServerCertificatesVolumeSource()
		}
	}
}

// setSelfHostedVolumesForControllerManager makes sure the self-hosted controller manager has the right volume source coming from a self-hosted cluster
func setSelfHostedVolumesForControllerManager(podSpec *v1.PodSpec) {
	for i, v := range podSpec.Volumes {
		// If the volume name matches the expected one; switch the volume source from hostPath to cluster-hosted
		if v.Name == kubeadmconstants.KubeCertificatesVolumeName {
			podSpec.Volumes[i].VolumeSource = controllerManagerCertificatesVolumeSource()
		} else if v.Name == kubeadmconstants.KubeConfigVolumeName {
			podSpec.Volumes[i].VolumeSource = kubeConfigVolumeSource(kubeadmconstants.ControllerManagerKubeConfigFileName)
		}
	}

	// Change directory for the kubeconfig directory to selfHostedKubeConfigDir
	for i, vm := range podSpec.Containers[0].VolumeMounts {
		if vm.Name == kubeadmconstants.KubeConfigVolumeName {
			podSpec.Containers[0].VolumeMounts[i].MountPath = selfHostedKubeConfigDir
		}
	}

	// Rewrite the --kubeconfig path as the volume mount path may not overlap with certs dir, which it does by default (/etc/kubernetes and /etc/kubernetes/pki)
	// This is not a problem with hostPath mounts as hostPath supports mounting one file only, instead of always a full directory. Secrets and Projected Volumes
	// don't support that.
	podSpec.Containers[0].Command = kubeadmutil.ReplaceArgument(podSpec.Containers[0].Command, func(argMap map[string]string) map[string]string {
		controllerManagerKubeConfigPath := filepath.Join(selfHostedKubeConfigDir, kubeadmconstants.ControllerManagerKubeConfigFileName)
		argMap["kubeconfig"] = controllerManagerKubeConfigPath
		if _, ok := argMap["authentication-kubeconfig"]; ok {
			argMap["authentication-kubeconfig"] = controllerManagerKubeConfigPath
		}
		if _, ok := argMap["authorization-kubeconfig"]; ok {
			argMap["authorization-kubeconfig"] = controllerManagerKubeConfigPath
		}
		return argMap
	})
}

// setSelfHostedVolumesForScheduler makes sure the self-hosted scheduler has the right volume source coming from a self-hosted cluster
func setSelfHostedVolumesForScheduler(podSpec *v1.PodSpec) {
	for i, v := range podSpec.Volumes {
		// If the volume name matches the expected one; switch the volume source from hostPath to cluster-hosted
		if v.Name == kubeadmconstants.KubeConfigVolumeName {
			podSpec.Volumes[i].VolumeSource = kubeConfigVolumeSource(kubeadmconstants.SchedulerKubeConfigFileName)
		}
	}

	// Change directory for the kubeconfig directory to selfHostedKubeConfigDir
	for i, vm := range podSpec.Containers[0].VolumeMounts {
		if vm.Name == kubeadmconstants.KubeConfigVolumeName {
			podSpec.Containers[0].VolumeMounts[i].MountPath = selfHostedKubeConfigDir
		}
	}

	// Rewrite the --kubeconfig path as the volume mount path may not overlap with certs dir, which it does by default (/etc/kubernetes and /etc/kubernetes/pki)
	// This is not a problem with hostPath mounts as hostPath supports mounting one file only, instead of always a full directory. Secrets and Projected Volumes
	// don't support that.
	podSpec.Containers[0].Command = kubeadmutil.ReplaceArgument(podSpec.Containers[0].Command, func(argMap map[string]string) map[string]string {
		schedulerKubeConfigPath := filepath.Join(selfHostedKubeConfigDir, kubeadmconstants.SchedulerKubeConfigFileName)
		argMap["kubeconfig"] = schedulerKubeConfigPath
		if _, ok := argMap["authentication-kubeconfig"]; ok {
			argMap["authentication-kubeconfig"] = schedulerKubeConfigPath
		}
		if _, ok := argMap["authorization-kubeconfig"]; ok {
			argMap["authorization-kubeconfig"] = schedulerKubeConfigPath
		}
		return argMap
	})
}
