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
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/exec"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
)

// Result holds the execution result of remote execution command.
type Result struct {
	Host   string
	Cmd    string
	Stdout string
	Stderr string
	Code   int
}

// LogResult records result log
func LogResult(result Result) {
	remote := result.Host
	framework.Logf("exec %s: command:   %s", remote, result.Cmd)
	framework.Logf("exec %s: stdout:    %q", remote, result.Stdout)
	framework.Logf("exec %s: stderr:    %q", remote, result.Stderr)
	framework.Logf("exec %s: exit code: %d", remote, result.Code)
}

// HostExec represents interface we require to execute commands on remote host.
type HostExec interface {
	Execute(ctx context.Context, cmd string, node *v1.Node) (Result, error)
	IssueCommandWithResult(ctx context.Context, cmd string, node *v1.Node) (string, error)
	IssueCommand(ctx context.Context, cmd string, node *v1.Node) error
	Cleanup(ctx context.Context)
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
func (h *hostExecutor) launchNodeExecPod(ctx context.Context, node string) *v1.Pod {
	f := h.Framework
	cs := f.ClientSet
	ns := f.Namespace

	hostExecPod := e2epod.NewExecPodSpec(ns.Name, "", true)
	hostExecPod.GenerateName = fmt.Sprintf("hostexec-%s-", node)

	if framework.TestContext.NodeE2E {
		// E2E node tests do not run a scheduler, so set the node name directly
		hostExecPod.Spec.NodeName = node
	} else {
		// Use NodeAffinity instead of NodeName so that pods will not
		// be immediately Failed by kubelet if it's out of space. Instead
		// Pods will be pending in the scheduler until there is space freed
		// up.
		e2epod.SetNodeAffinity(&hostExecPod.Spec, node)

	}
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
	pod, err := cs.CoreV1().Pods(ns.Name).Create(ctx, hostExecPod, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	err = e2epod.WaitTimeoutForPodRunningInNamespace(ctx, cs, pod.Name, pod.Namespace, f.Timeouts.PodStart)
	framework.ExpectNoError(err)
	return pod
}

// Execute executes the command on the given node. If there is no error
// performing the remote command execution, the stdout, stderr and exit code
// are returned.
// This works like ssh.SSH(...) utility.
func (h *hostExecutor) Execute(ctx context.Context, cmd string, node *v1.Node) (Result, error) {
	result, err := h.exec(ctx, cmd, node)
	if codeExitErr, ok := err.(exec.CodeExitError); ok {
		// extract the exit code of remote command and silence the command
		// non-zero exit code error
		result.Code = codeExitErr.ExitStatus()
		err = nil
	}
	return result, err
}

func (h *hostExecutor) exec(ctx context.Context, cmd string, node *v1.Node) (Result, error) {
	result := Result{
		Host: node.Name,
		Cmd:  cmd,
	}
	pod, ok := h.nodeExecPods[node.Name]
	if !ok {
		pod = h.launchNodeExecPod(ctx, node.Name)
		if pod == nil {
			return result, fmt.Errorf("failed to create hostexec pod for node %q", node)
		}
		h.nodeExecPods[node.Name] = pod
	}
	args := []string{
		"nsenter",
		"--mount=/rootfs/proc/1/ns/mnt",
		"--",
		"sh",
		"-c",
		cmd,
	}
	containerName := pod.Spec.Containers[0].Name
	var err error
	result.Stdout, result.Stderr, err = e2epod.ExecWithOptions(h.Framework, e2epod.ExecOptions{
		Command:            args,
		Namespace:          pod.Namespace,
		PodName:            pod.Name,
		ContainerName:      containerName,
		Stdin:              nil,
		CaptureStdout:      true,
		CaptureStderr:      true,
		PreserveWhitespace: true,
	})
	return result, err
}

// IssueCommandWithResult issues command on the given node and returns stdout as
// result. It returns error if there are some issues executing the command or
// the command exits non-zero.
func (h *hostExecutor) IssueCommandWithResult(ctx context.Context, cmd string, node *v1.Node) (string, error) {
	result, err := h.exec(ctx, cmd, node)
	if err != nil {
		LogResult(result)
	}
	return result.Stdout, err
}

// IssueCommand works like IssueCommandWithResult, but discards result.
func (h *hostExecutor) IssueCommand(ctx context.Context, cmd string, node *v1.Node) error {
	_, err := h.IssueCommandWithResult(ctx, cmd, node)
	return err
}

// Cleanup cleanup resources it created during test.
// Note that in most cases it is not necessary to call this because we create
// pods under test namespace which will be destroyed in teardown phase.
func (h *hostExecutor) Cleanup(ctx context.Context) {
	for _, pod := range h.nodeExecPods {
		e2epod.DeletePodOrFail(ctx, h.Framework.ClientSet, pod.Namespace, pod.Name)
	}
	h.nodeExecPods = make(map[string]*v1.Pod)
}
