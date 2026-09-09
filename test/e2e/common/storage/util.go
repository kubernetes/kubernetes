/*
Copyright 2016 The Kubernetes Authors.

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

package storage

import (
	"fmt"
	"os"

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/test/e2e/framework"
)

var (
	// non-Administrator Windows user used in tests. This is the Windows equivalent of the Linux non-root UID usage.
	nonAdminTestUserName = "ContainerUser"
	// non-root UID used in tests.
	nonRootTestUserID = int64(1000)
)

// setPodNonRootUser configures the Pod to run as a non-root user.
// For Windows, it sets the RunAsUserName field to ContainerUser, and for Linux, it sets the RunAsUser field to 1000.
func setPodNonRootUser(pod *v1.Pod) {
	if framework.NodeOSDistroIs("windows") {
		pod.Spec.SecurityContext.WindowsOptions = &v1.WindowsSecurityContextOptions{RunAsUserName: &nonAdminTestUserName}
	} else {
		pod.Spec.SecurityContext.RunAsUser = &nonRootTestUserID
	}
}

// getFileModeRegex returns a file mode related regex which should be matched by the mounttest pods' output.
// If the given mask is nil, then the regex will contain the default OS file modes, which are 0644 for Linux and 0775 for Windows.
func getFileModeRegex(filePath string, mask *int32) string {
	var (
		linuxMask   int32
		windowsMask int32
	)
	if mask == nil {
		linuxMask = int32(0644)
		windowsMask = int32(0775)
	} else {
		linuxMask = *mask
		windowsMask = *mask
	}

	linuxOutput := fmt.Sprintf("mode of file \"%s\": %v", filePath, os.FileMode(linuxMask))
	windowsOutput := fmt.Sprintf("mode of Windows file \"%v\": %s", filePath, os.FileMode(windowsMask))

	return fmt.Sprintf("(%s|%s)", linuxOutput, windowsOutput)
}

// createMounts creates a v1.VolumeMount list with a single element.
func createMounts(volumeName, volumeMountPath string, readOnly bool) []v1.VolumeMount {
	return []v1.VolumeMount{
		{
			Name:      volumeName,
			MountPath: volumeMountPath,
			ReadOnly:  readOnly,
		},
	}
}
