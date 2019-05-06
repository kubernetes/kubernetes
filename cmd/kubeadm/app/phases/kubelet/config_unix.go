// +build !windows

/*
Copyright 2019 The Kubernetes Authors.

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

package kubelet

import (
	"k8s.io/klog"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/util/procfs"
	utilsexec "k8s.io/utils/exec"
)

// buildDynamicDefaults returns a Kubelet config file with corrected values according to different OS configuration
func buildDynamicDefaults(kubeletConfig *kubeletconfig.KubeletConfiguration, nodeReg *kubeadmapi.NodeRegistrationOptions) *kubeletconfig.KubeletConfiguration {

	if nodeReg.CRISocket == kubeadmconstants.DefaultDockerCRISocket {
		driver, err := kubeadmutil.GetCgroupDriverDocker(utilsexec.New())
		if err != nil {
			klog.Warningf("cannot automatically assign a cgroupDriver value in the KubeletConfiguration file: %v\n", err)
		} else {
			kubeletConfig.CgroupDriver = driver
		}
	}

	// TODO: Conditionally set `CgroupDriver` to either `systemd` or `cgroupfs` for CRI other than Docker

	if pids, _ := procfs.PidOf("systemd-resolved"); len(pids) > 0 {
		// procfs.PidOf only returns an error if the regex is empty or doesn't compile, so we can ignore it
		kubeletConfig.ResolverConfig = "/run/systemd/resolve/resolv.conf"
	}

	return kubeletConfig
}
