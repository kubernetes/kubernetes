//go:build windows
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
	"context"
	"fmt"
	"strings"

	"k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/securitycontext"
)

var (
	windowsRootUserName = "ContainerAdministrator"
)

// verifyRunAsNonRoot verifies RunAsNonRoot on windows.
// https://github.com/kubernetes/enhancements/tree/master/keps/sig-windows/116-windows-node-support#v1container
// Windows does not have a root user, we cannot judge the root identity of the windows container by the way to judge the root(uid=0) of the linux container.
// According to the discussion of sig-windows, at present, we assume that ContainerAdministrator is the windows container root user,
// and then optimize this logic according to the best time.
// https://docs.google.com/document/d/1Tjxzjjuy4SQsFSUVXZbvqVb64hjNAG5CQX8bK7Yda9w
// note: usernames on Windows are NOT case sensitive!
func verifyRunAsNonRoot(ctx context.Context, pod *v1.Pod, container *v1.Container, uid *int64, username string) error {
	logger := klog.FromContext(ctx)
	effectiveSc := securitycontext.DetermineEffectiveSecurityContext(pod, container)
	// If the option is not set, or if running as root is allowed, return nil.
	if effectiveSc == nil || effectiveSc.RunAsNonRoot == nil || !*effectiveSc.RunAsNonRoot {
		return nil
	}
	if effectiveSc.RunAsUser != nil {
		logger.Info("Windows container does not support SecurityContext.RunAsUser, please use SecurityContext.WindowsOptions",
			"pod", klog.KObj(pod), "containerName", container.Name)
	}
	if effectiveSc.SELinuxOptions != nil {
		logger.Info("Windows container does not support SecurityContext.SELinuxOptions, please use SecurityContext.WindowsOptions",
			"pod", klog.KObj(pod), "containerName", container.Name)
	}
	if effectiveSc.RunAsGroup != nil {
		logger.Info("Windows container does not support SecurityContext.RunAsGroup", "pod", klog.KObj(pod), "containerName", container.Name)
	}
	// Verify that if runAsUserName is set for the pod and/or container that it is not set to 'ContainerAdministrator'
	if effectiveSc.WindowsOptions != nil {
		if effectiveSc.WindowsOptions.RunAsUserName != nil {
			if strings.EqualFold(*effectiveSc.WindowsOptions.RunAsUserName, windowsRootUserName) {
				return fmt.Errorf("container's runAsUserName (%s) which will be regarded as root identity and will break non-root policy (pod: %q, container: %s)", *effectiveSc.WindowsOptions.RunAsUserName, format.Pod(pod), container.Name)
			}
			return nil
		}
	}
	// Verify that if runAsUserName is NOT set for the pod and/or container that the default user for the container image is not set to 'ContainerAdministrator'
	if len(username) > 0 && strings.EqualFold(username, windowsRootUserName) {
		return fmt.Errorf("container's runAsUser (%s) which will be regarded as root identity and will break non-root policy (pod: %q, container: %s)", username, format.Pod(pod), container.Name)
	}
	return nil
}
