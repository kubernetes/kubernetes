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

package pod

import (
	"bytes"
	"context"
	"io"
	"net/url"
	"strings"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/scheme"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/remotecommand"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/onsi/gomega"
)

// ExecOptions passed to ExecWithOptions
type ExecOptions struct {
	Command       []string
	Namespace     string
	PodName       string
	ContainerName string
	Stdin         io.Reader
	CaptureStdout bool
	CaptureStderr bool
	// If false, whitespace in std{err,out} will be removed.
	PreserveWhitespace bool
	Quiet              bool
}

// ExecWithOptions executes a command in the specified container,
// returning stdout, stderr and error. `options` allowed for
// additional parameters to be passed.
func ExecWithOptions(f *framework.Framework, options ExecOptions) (string, string, error) {
	return ExecWithOptionsContext(context.Background(), f, options)
}

func ExecWithOptionsContext(ctx context.Context, f *framework.Framework, options ExecOptions) (string, string, error) {
	if !options.Quiet {
		framework.Logf("ExecWithOptions %+v", options)
	}
	config, err := framework.LoadConfig()
	framework.ExpectNoError(err, "failed to load restclient config")

	const tty = false

	framework.Logf("ExecWithOptions: Clientset creation")
	req := f.ClientSet.CoreV1().RESTClient().Post().
		Resource("pods").
		Name(options.PodName).
		Namespace(options.Namespace).
		SubResource("exec").
		Param("container", options.ContainerName)
	req.VersionedParams(&v1.PodExecOptions{
		Container: options.ContainerName,
		Command:   options.Command,
		Stdin:     options.Stdin != nil,
		Stdout:    options.CaptureStdout,
		Stderr:    options.CaptureStderr,
		TTY:       tty,
	}, scheme.ParameterCodec)

	var stdout, stderr bytes.Buffer
	framework.Logf("ExecWithOptions: execute(POST %s)", req.URL())
	err = execute(ctx, "POST", req.URL(), config, options.Stdin, &stdout, &stderr, tty)

	if options.PreserveWhitespace {
		return stdout.String(), stderr.String(), err
	}
	return strings.TrimSpace(stdout.String()), strings.TrimSpace(stderr.String()), err
}

// ExecCommandInContainerWithFullOutput executes a command in the
// specified container and return stdout, stderr and error
func ExecCommandInContainerWithFullOutput(f *framework.Framework, podName, containerName string, cmd ...string) (string, string, error) {
	// TODO (pohly): add context support
	return ExecWithOptions(f, ExecOptions{
		Command:            cmd,
		Namespace:          f.Namespace.Name,
		PodName:            podName,
		ContainerName:      containerName,
		Stdin:              nil,
		CaptureStdout:      true,
		CaptureStderr:      true,
		PreserveWhitespace: false,
	})
}

// ExecCommandInContainer executes a command in the specified container.
func ExecCommandInContainer(f *framework.Framework, podName, containerName string, cmd ...string) string {
	stdout, stderr, err := ExecCommandInContainerWithFullOutput(f, podName, containerName, cmd...)
	framework.Logf("Exec stderr: %q", stderr)
	framework.ExpectNoError(err,
		"failed to execute command in pod %v, container %v: %v",
		podName, containerName, err)
	return stdout
}

// ExecShellInContainer executes the specified command on the pod's container.
func ExecShellInContainer(f *framework.Framework, podName, containerName string, cmd string) string {
	return ExecCommandInContainer(f, podName, containerName, "/bin/sh", "-c", cmd)
}

func execCommandInPod(ctx context.Context, f *framework.Framework, podName string, cmd ...string) string {
	pod, err := NewPodClient(f).Get(ctx, podName, metav1.GetOptions{})
	framework.ExpectNoError(err, "failed to get pod %v", podName)
	gomega.Expect(pod.Spec.Containers).NotTo(gomega.BeEmpty())
	return ExecCommandInContainer(f, podName, pod.Spec.Containers[0].Name, cmd...)
}

func execCommandInPodWithFullOutput(ctx context.Context, f *framework.Framework, podName string, cmd ...string) (string, string, error) {
	pod, err := NewPodClient(f).Get(ctx, podName, metav1.GetOptions{})
	framework.ExpectNoError(err, "failed to get pod %v", podName)
	gomega.Expect(pod.Spec.Containers).NotTo(gomega.BeEmpty())
	return ExecCommandInContainerWithFullOutput(f, podName, pod.Spec.Containers[0].Name, cmd...)
}

// ExecShellInPod executes the specified command on the pod.
func ExecShellInPod(ctx context.Context, f *framework.Framework, podName string, cmd string) string {
	return execCommandInPod(ctx, f, podName, "/bin/sh", "-c", cmd)
}

// ExecShellInPodWithFullOutput executes the specified command on the Pod and returns stdout, stderr and error.
func ExecShellInPodWithFullOutput(ctx context.Context, f *framework.Framework, podName string, cmd string) (string, string, error) {
	return execCommandInPodWithFullOutput(ctx, f, podName, "/bin/sh", "-c", cmd)
}

func execute(ctx context.Context, method string, url *url.URL, config *restclient.Config, stdin io.Reader, stdout, stderr io.Writer, tty bool) error {
	exec, err := remotecommand.NewSPDYExecutor(config, method, url)
	if err != nil {
		return err
	}
	return exec.StreamWithContext(ctx, remotecommand.StreamOptions{
		Stdin:  stdin,
		Stdout: stdout,
		Stderr: stderr,
		Tty:    tty,
	})
}
