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

package kubelet

import (
	"fmt"

	"k8s.io/klog/v2"

	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/initsystem"
)

// TryStartKubelet attempts to bring up kubelet service
func TryStartKubelet() {
	// If we notice that the kubelet service is inactive, try to start it
	initSystem, err := initsystem.GetInitSystem()
	if err != nil {
		fmt.Println("[kubelet-start] No supported init system detected, won't make sure the kubelet is running properly.")
		return
	}

	if !initSystem.ServiceExists(kubeadmconstants.Kubelet) {
		fmt.Println("[kubelet-start] Couldn't detect a kubelet service, can't make sure the kubelet is running properly.")
	}

	// This runs "systemctl daemon-reload && systemctl restart kubelet"
	if err := initSystem.ServiceRestart(kubeadmconstants.Kubelet); err != nil {
		klog.Warningf("[kubelet-start] WARNING: unable to start the kubelet service: [%v]\n", err)
		fmt.Printf("[kubelet-start] Please ensure kubelet is reloaded and running manually.\n")
	}
}

// TryStopKubelet attempts to bring down the kubelet service momentarily
func TryStopKubelet() {
	// If we notice that the kubelet service is inactive, try to start it
	initSystem, err := initsystem.GetInitSystem()
	if err != nil {
		fmt.Println("[kubelet-start] No supported init system detected, won't make sure the kubelet not running for a short period of time while setting up configuration for it.")
		return
	}

	if !initSystem.ServiceExists(kubeadmconstants.Kubelet) {
		fmt.Println("[kubelet-start] Couldn't detect a kubelet service, can't make sure the kubelet not running for a short period of time while setting up configuration for it.")
	}

	// This runs "systemctl daemon-reload && systemctl stop kubelet"
	if err := initSystem.ServiceStop(kubeadmconstants.Kubelet); err != nil {
		klog.Warningf("[kubelet-start] WARNING: unable to stop the kubelet service momentarily: [%v]\n", err)
	}
}

// TryRestartKubelet attempts to restart the kubelet service
func TryRestartKubelet() {
	// If we notice that the kubelet service is inactive, try to start it
	initSystem, err := initsystem.GetInitSystem()
	if err != nil {
		fmt.Println("[kubelet-start] No supported init system detected, won't make sure the kubelet not running for a short period of time while setting up configuration for it.")
		return
	}

	if !initSystem.ServiceExists(kubeadmconstants.Kubelet) {
		fmt.Println("[kubelet-start] Couldn't detect a kubelet service, can't make sure the kubelet not running for a short period of time while setting up configuration for it.")
	}

	// This runs "systemctl daemon-reload && systemctl stop kubelet"
	if err := initSystem.ServiceRestart(kubeadmconstants.Kubelet); err != nil {
		klog.Warningf("[kubelet-start] WARNING: unable to restart the kubelet service momentarily: [%v]\n", err)
	}
}
