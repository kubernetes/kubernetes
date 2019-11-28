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

package options

import (
	"runtime"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/kubelet/config"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

const (
	// When these values are updated, also update test/e2e/framework/util.go
	defaultPodSandboxImageName    = "k8s.gcr.io/pause"
	defaultPodSandboxImageVersion = "3.1"
	defaultPodSandboxSeccomp      = `{
	"defaultAction":"SCMP_ACT_ERRNO",
	"archMap":[{
		"architecture":"SCMP_ARCH_X86_64","subArchitectures":["SCMP_ARCH_X86","SCMP_ARCH_X32"] },{
		"architecture":"SCMP_ARCH_AARCH64","subArchitectures":["SCMP_ARCH_ARM"] },{
		"architecture":"SCMP_ARCH_MIPS64","subArchitectures":["SCMP_ARCH_MIPS","SCMP_ARCH_MIPS64N32"] },{
		"architecture":"SCMP_ARCH_MIPS64N32","subArchitectures":["SCMP_ARCH_MIPS","SCMP_ARCH_MIPS64"] },{
		"architecture":"SCMP_ARCH_MIPSEL64","subArchitectures":["SCMP_ARCH_MIPSEL","SCMP_ARCH_MIPSEL64N32"] },{
		"architecture":"SCMP_ARCH_MIPSEL64N32","subArchitectures":["SCMP_ARCH_MIPSEL","SCMP_ARCH_MIPSEL64"] },{
		"architecture":"SCMP_ARCH_S390X","subArchitectures":["SCMP_ARCH_S390"] 
	}],
	"syscalls":[{
		"names":["arch_prctl","brk","close","execve","exit_group","futex","mprotect","nanosleep","stat","write","uname","pause","rt_sigaction","getpid","wait4","waitid","fork","getppid"],
		"action":"SCMP_ACT_ALLOW"
	}]
}`
)

var (
	defaultPodSandboxImage = defaultPodSandboxImageName +
		":" + defaultPodSandboxImageVersion
)

// NewContainerRuntimeOptions will create a new ContainerRuntimeOptions with
// default values.
func NewContainerRuntimeOptions() *config.ContainerRuntimeOptions {
	dockerEndpoint := ""
	if runtime.GOOS != "windows" {
		dockerEndpoint = "unix:///var/run/docker.sock"
	}

	return &config.ContainerRuntimeOptions{
		ContainerRuntime:           kubetypes.DockerContainerRuntime,
		RedirectContainerStreaming: false,
		DockerEndpoint:             dockerEndpoint,
		DockershimRootDirectory:    "/var/lib/dockershim",
		PodSandboxImage:            defaultPodSandboxImage,
		PodSandboxSeccomp:          defaultPodSandboxSeccomp,
		ImagePullProgressDeadline:  metav1.Duration{Duration: 1 * time.Minute},
		ExperimentalDockershim:     false,

		//Alpha feature
		CNIBinDir:   "/opt/cni/bin",
		CNIConfDir:  "/etc/cni/net.d",
		CNICacheDir: "/var/lib/cni/cache",
	}
}
