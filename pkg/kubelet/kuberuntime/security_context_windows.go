// +build windows

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

package kuberuntime

import (
	"fmt"
	"k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/securitycontext"
)

var (
	windowsRootUserName = "ContainerAdministrator"
)

// verifyRunAsNonRoot verifies RunAsNonRoot on windows.
// https://github.com/kubernetes/enhancements/blob/master/keps/sig-windows/20190103-windows-node-support.md#v1container
// Windows does not have a root user, we cannot judge the root identity of the windows container by the way to judge the root(uid=0) of the linux container.
// According to the discussion of sig-windows, at present, we assume that ContainerAdministrator is the windows container root user,
// and then optimize this logic according to the best time.
// https://docs.google.com/document/d/1Tjxzjjuy4SQsFSUVXZbvqVb64hjNAG5CQX8bK7Yda9w
func verifyRunAsNonRoot(pod *v1.Pod, container *v1.Container, uid *int64, username string) error {
	effectiveSc := securitycontext.DetermineEffectiveSecurityContext(pod, container)
	// If the option is not set, or if running as root is allowed, return nil.
	if effectiveSc == nil || effectiveSc.RunAsNonRoot == nil || !*effectiveSc.RunAsNonRoot {
		return nil
	}
	if effectiveSc.RunAsUser != nil {
		klog.Warningf("Windows container does not support SecurityContext.RunAsUser, please use SecurityContext.WindowsOptions (pod: %q, container: %s)", format.Pod(pod), container.Name)
	}
	if effectiveSc.SELinuxOptions != nil {
		klog.Warningf("Windows container does not support SecurityContext.SELinuxOptions, please use SecurityContext.WindowsOptions (pod: %q, container: %s)", format.Pod(pod), container.Name)
	}
	if effectiveSc.RunAsGroup != nil {
		klog.Warningf("Windows container does not support SecurityContext.RunAsGroup (pod: %q, container: %s)", format.Pod(pod), container.Name)
	}
	if effectiveSc.WindowsOptions != nil {
		if effectiveSc.WindowsOptions.RunAsUserName != nil {
			if *effectiveSc.WindowsOptions.RunAsUserName == windowsRootUserName {
				return fmt.Errorf("container's runAsUser (%s) which will be regarded as root identity and will break non-root policy (pod: %q, container: %s)", username, format.Pod(pod), container.Name)
			}
			return nil
		}
	}
	if len(username) > 0 && username == windowsRootUserName {
		return fmt.Errorf("container's runAsUser (%s) which will be regarded as root identity and will break non-root policy (pod: %q, container: %s)", username, format.Pod(pod), container.Name)
	}
	return nil
}
