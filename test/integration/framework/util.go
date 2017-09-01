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

// TODO: This file can potentially be moved to a common place used by both e2e and integration tests.

package framework

import (
	"strings"

	clientset "k8s.io/client-go/kubernetes"
)

const (
	// When these values are updated, also update cmd/kubelet/app/options/options.go
	// A copy of these values exist in e2e/framework/util.go.
	currentPodInfraContainerImageName    = "gcr.io/google_containers/pause"
	currentPodInfraContainerImageVersion = "3.0"
)

// GetServerArchitecture fetches the architecture of the cluster's apiserver.
func GetServerArchitecture(c clientset.Interface) string {
	arch := ""
	sVer, err := c.Discovery().ServerVersion()
	if err != nil || sVer.Platform == "" {
		// If we failed to get the server version for some reason, default to amd64.
		arch = "amd64"
	} else {
		// Split the platform string into OS and Arch separately.
		// The platform string may for example be "linux/amd64", "linux/arm" or "windows/amd64".
		osArchArray := strings.Split(sVer.Platform, "/")
		arch = osArchArray[1]
	}
	return arch
}

// GetPauseImageName fetches the pause image name for the same architecture as the apiserver.
func GetPauseImageName(c clientset.Interface) string {
	return currentPodInfraContainerImageName + "-" + GetServerArchitecture(c) + ":" + currentPodInfraContainerImageVersion
}
