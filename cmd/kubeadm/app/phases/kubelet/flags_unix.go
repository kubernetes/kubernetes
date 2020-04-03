// +build !windows

/*
Copyright 2020 The Kubernetes Authors.

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
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

// buildKubeletArgMap takes a kubeletFlagsOpts object and builds based on that a string-string map with flags
// that should be given to the local Linux kubelet daemon.
func buildKubeletArgMap(opts kubeletFlagsOpts) map[string]string {
	kubeletFlags := buildKubeletArgMapCommon(opts)

	// TODO: Conditionally set `--cgroup-driver` to either `systemd` or `cgroupfs` for CRI other than Docker
	if opts.nodeRegOpts.CRISocket == constants.DefaultDockerCRISocket {
		driver, err := kubeadmutil.GetCgroupDriverDocker(opts.execer)
		if err != nil {
			klog.Warningf("cannot automatically assign a '--cgroup-driver' value when starting the Kubelet: %v\n", err)
		} else {
			kubeletFlags["cgroup-driver"] = driver
		}
	}

	ok, err := opts.isServiceActiveFunc("systemd-resolved")
	if err != nil {
		klog.Warningf("cannot determine if systemd-resolved is active: %v\n", err)
	}
	if ok {
		kubeletFlags["resolv-conf"] = "/run/systemd/resolve/resolv.conf"
	}

	return kubeletFlags
}
