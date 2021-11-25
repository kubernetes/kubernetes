/*
Copyright 2021 The Kubernetes Authors.

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

package pod

import (
	"flag"

	v1 "k8s.io/api/core/v1"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

// NodeOSDistroIs returns true if the distro is the same as `--node-os-distro`
// the package framework/pod can't import the framework package (see #81245)
// we need to check if the --node-os-distro=windows is set and the framework package
// is the one that's parsing the flags, as a workaround this method is looking for the same flag again
// TODO: replace with `framework.NodeOSDistroIs` when #81245 is complete
func NodeOSDistroIs(distro string) bool {
	var nodeOsDistro *flag.Flag = flag.Lookup("node-os-distro")
	if nodeOsDistro != nil && nodeOsDistro.Value.String() == distro {
		return true
	}
	return false
}

// GenerateScriptCmd generates the corresponding command lines to execute a command.
func GenerateScriptCmd(command string) []string {
	var commands []string
	commands = []string{"/bin/sh", "-c", command}
	return commands
}

// GetDefaultTestImage returns the default test image based on OS.
// If the node OS is windows, currently we return Agnhost image for Windows node
// due to the issue of #https://github.com/kubernetes-sigs/windows-testing/pull/35.
// If the node OS is linux, return busybox image
func GetDefaultTestImage() string {
	return imageutils.GetE2EImage(GetDefaultTestImageID())
}

// GetDefaultTestImageID returns the default test image id based on OS.
// If the node OS is windows, currently we return Agnhost image for Windows node
// due to the issue of #https://github.com/kubernetes-sigs/windows-testing/pull/35.
// If the node OS is linux, return busybox image
func GetDefaultTestImageID() int {
	return GetTestImageID(imageutils.BusyBox)
}

// GetTestImage returns the image name with the given input
// If the Node OS is windows, currently we return Agnhost image for Windows node
// due to the issue of #https://github.com/kubernetes-sigs/windows-testing/pull/35.
func GetTestImage(id int) string {
	if NodeOSDistroIs("windows") {
		return imageutils.GetE2EImage(imageutils.Agnhost)
	}
	return imageutils.GetE2EImage(id)
}

// GetTestImageID returns the image id with the given input
// If the Node OS is windows, currently we return Agnhost image for Windows node
// due to the issue of #https://github.com/kubernetes-sigs/windows-testing/pull/35.
func GetTestImageID(id int) int {
	if NodeOSDistroIs("windows") {
		return imageutils.Agnhost
	}
	return id
}

// GeneratePodSecurityContext generates the corresponding pod security context with the given inputs
// If the Node OS is windows, currently we will ignore the inputs and return nil.
// TODO: Will modify it after windows has its own security context
func GeneratePodSecurityContext(fsGroup *int64, seLinuxOptions *v1.SELinuxOptions) *v1.PodSecurityContext {
	if NodeOSDistroIs("windows") {
		return nil
	}
	return &v1.PodSecurityContext{
		FSGroup:        fsGroup,
		SELinuxOptions: seLinuxOptions,
	}
}

// GenerateContainerSecurityContext generates the corresponding container security context with the given inputs
// If the Node OS is windows, currently we will ignore the inputs and return nil.
// TODO: Will modify it after windows has its own security context
func GenerateContainerSecurityContext(privileged bool) *v1.SecurityContext {
	if NodeOSDistroIs("windows") {
		return nil
	}
	return &v1.SecurityContext{
		Privileged: &privileged,
	}
}

// GetLinuxLabel returns the default SELinuxLabel based on OS.
// If the node OS is windows, it will return nil
func GetLinuxLabel() *v1.SELinuxOptions {
	if NodeOSDistroIs("windows") {
		return nil
	}
	return &v1.SELinuxOptions{
		Level: "s0:c0,c1"}
}
