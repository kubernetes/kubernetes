//go:build windows

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

package validation

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"

	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

// validateKubeletOSConfiguration validates os specific kubelet configuration and returns an error if it is invalid.
func validateKubeletOSConfiguration(kc *kubeletconfig.KubeletConfiguration) error {
	message := "ignored configuration option: %v (%v) %v is not supported on Windows"

	if kc.CgroupsPerQOS {
		klog.Warningf(message, "CgroupsPerQOS", "--cgroups-per-qos", kc.CgroupsPerQOS)
	}

	// Inode-based eviction signals cannot be enforced on Windows: NTFS has no POSIX
	// inodes and winstats.GetDirFsInfo leaves the inode counters nil, so the eviction
	// manager never creates an observation for them and they would never fire. Rather
	// than failing kubelet startup (which would break otherwise-valid shared configs)
	// or silently accepting them (which misleads operators into thinking inode eviction
	// is active), warn and drop them. This mirrors how containerfs.inodesFree is handled
	// in pkg/kubelet/eviction/helpers.go.
	for _, v := range []struct {
		name string
		flag string
	}{{
		name: "EvictionHard",
		flag: "--eviction-hard",
	}, {
		name: "EvictionSoft",
		flag: "--eviction-soft",
	}, {
		name: "EvictionMinimumReclaim",
		flag: "--eviction-minimum-reclaim",
	}} {
		m := map[string]string{}
		switch v.name {
		case "EvictionHard":
			m = kc.EvictionHard
		case "EvictionSoft":
			m = kc.EvictionSoft
		case "EvictionMinimumReclaim":
			m = kc.EvictionMinimumReclaim
		}
		for _, signal := range []evictionapi.Signal{
			evictionapi.SignalNodeFsInodesFree,
			evictionapi.SignalImageFsInodesFree,
			evictionapi.SignalContainerFsInodesFree,
		} {
			if _, ok := m[string(signal)]; ok {
				klog.Warningf("ignoring configuration option: inode eviction signal %s (%s) is not supported on Windows and will be ignored", signal, v.flag)
				delete(m, string(signal))
			}
		}
	}

	if kc.SingleProcessOOMKill != nil {
		return fmt.Errorf("invalid configuration: singleProcessOOMKill is not supported on Windows")
	}

	enforceNodeAllocatableWithoutNone := sets.New(kc.EnforceNodeAllocatable...).Delete(kubetypes.NodeAllocatableNoneKey)
	if len(enforceNodeAllocatableWithoutNone) > 0 {
		klog.Warningf(message, "EnforceNodeAllocatable", "--enforce-node-allocatable", kc.EnforceNodeAllocatable)
	}

	if kc.UserNamespaces != nil {
		return fmt.Errorf("invalid configuration: userNamespaces is not supported on Windows")
	}

	if len(kc.DefaultPodSysctls) > 0 {
		return fmt.Errorf("invalid configuration: defaultPodSysctls is not supported on Windows")
	}

	return nil
}
