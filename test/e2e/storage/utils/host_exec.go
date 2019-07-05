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

package utils

import (
	"fmt"

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
)

// HostExec represents interface we require to execute commands on remote host.
type HostExec interface {
	IssueCommandWithResult(cmd string, node *v1.Node) (string, error)
	IssueCommand(cmd string, node *v1.Node) error
	Cleanup()
}

// hostExecutor implements HostExec
type hostExecutor struct {
	*framework.Framework
	nodeExecPods map[string]*v1.Pod
}

// NewHostExec returns a HostExec
func NewHostExec(framework *framework.Framework) HostExec {
	return &hostExecutor{
		Framework:    framework,
		nodeExecPods: make(map[string]*v1.Pod),
	}
}

// launchNodeExecPod launches a hostexec pod for local PV and waits
// until it's Running.
func (h *hostExecutor) launchNodeExecPod(node string) *v1.Pod {
	f := h.Framework
	cs := f.ClientSet
	ns := f.Namespace
	hostExecPod := e2epod.NewExecPodSpec(ns.Name, fmt.Sprintf("hostexec-%s", node), true)
	hostExecPod.Spec.NodeName = node
	hostExecPod.Spec.Volumes = []v1.Volume{
		{
			// Required to enter into host mount namespace via nsenter.
			Name: "rootfs",
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: "/",
				},
			},
		},
	}
	hostExecPod.Spec.Containers[0].VolumeMounts = []v1.VolumeMount{
		{
			Name:      "rootfs",
			MountPath: "/rootfs",
			ReadOnly:  true,
		},
	}
	hostExecPod.Spec.Containers[0].SecurityContext = &v1.SecurityContext{
		Privileged: func(privileged bool) *bool {
			return &privileged
		}(true),
	}
	pod, err := cs.CoreV1().Pods(ns.Name).Create(hostExecPod)
	framework.ExpectNoError(err)
	err = e2epod.WaitForPodRunningInNamespace(cs, pod)
	framework.ExpectNoError(err)
	return pod
}

// IssueCommandWithResult issues command on given node and returns stdout.
func (h *hostExecutor) IssueCommandWithResult(cmd string, node *v1.Node) (string, error) {
	pod, ok := h.nodeExecPods[node.Name]
	if !ok {
		pod = h.launchNodeExecPod(node.Name)
		if pod == nil {
			return "", fmt.Errorf("failed to create hostexec pod for node %q", node)
		}
		h.nodeExecPods[node.Name] = pod
	}
	args := []string{
		"exec",
		fmt.Sprintf("--namespace=%v", pod.Namespace),
		pod.Name,
		"--",
		"nsenter",
		"--mount=/rootfs/proc/1/ns/mnt",
		"--",
		"sh",
		"-c",
		cmd,
	}
	return framework.RunKubectl(args...)
}

// IssueCommand works like IssueCommandWithResult, but discards result.
func (h *hostExecutor) IssueCommand(cmd string, node *v1.Node) error {
	_, err := h.IssueCommandWithResult(cmd, node)
	return err
}

// Cleanup cleanup resources it created during test.
// Note that in most cases it is not necessary to call this because we create
// pods under test namespace which will be destroyed in teardown phase.
func (h *hostExecutor) Cleanup() {
	for _, pod := range h.nodeExecPods {
		e2epod.DeletePodOrFail(h.Framework.ClientSet, pod.Namespace, pod.Name)
	}
	h.nodeExecPods = make(map[string]*v1.Pod)
}
