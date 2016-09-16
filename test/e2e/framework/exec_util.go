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

package framework

import (
	"bytes"
	"io"
	"net/url"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/unversioned/remotecommand"
	remotecommandserver "k8s.io/kubernetes/pkg/kubelet/server/remotecommand"

	. "github.com/onsi/gomega"
)

// ExecCommandInContainer execute a command in the specified container.
// Pass in stdin, tty if needed in the future.
func (f *Framework) ExecCommandInContainer(podName, containerName string, cmd ...string) string {
	Logf("Exec running '%s'", strings.Join(cmd, " "))
	config, err := LoadConfig()
	Expect(err).NotTo(HaveOccurred(), "failed to load restclient config")
	var stdout, stderr bytes.Buffer
	var stdin io.Reader
	tty := false
	req := f.Client.RESTClient.Post().
		Resource("pods").
		Name(podName).
		Namespace(f.Namespace.Name).
		SubResource("exec").
		Param("container", containerName)
	req.VersionedParams(&api.PodExecOptions{
		Container: containerName,
		Command:   cmd,
		Stdin:     stdin != nil,
		Stdout:    true,
		Stderr:    true,
		TTY:       tty,
	}, api.ParameterCodec)

	err = execute("POST", req.URL(), config, stdin, &stdout, &stderr, tty)
	Expect(err).NotTo(HaveOccurred(), "post request failed")
	Logf("Exec stderr: %q", stderr.String())
	return strings.TrimSpace(stdout.String())
}

func (f *Framework) ExecShellInContainer(podName, containerName string, cmd string) string {
	return f.ExecCommandInContainer(podName, containerName, "/bin/sh", "-c", cmd)
}

func (f *Framework) ExecCommandInPod(podName string, cmd ...string) string {
	pod, err := f.PodClient().Get(podName)
	Expect(err).NotTo(HaveOccurred(), "failed to get pod")
	Expect(pod.Spec.Containers).NotTo(BeEmpty())
	return f.ExecCommandInContainer(podName, pod.Spec.Containers[0].Name, cmd...)
}

func (f *Framework) ExecShellInPod(podName string, cmd string) string {
	return f.ExecCommandInPod(podName, "/bin/sh", "-c", cmd)
}

func execute(method string, url *url.URL, config *restclient.Config, stdin io.Reader, stdout, stderr io.Writer, tty bool) error {
	exec, err := remotecommand.NewExecutor(config, method, url)
	if err != nil {
		return err
	}
	return exec.Stream(remotecommand.StreamOptions{
		SupportedProtocols: remotecommandserver.SupportedStreamingProtocols,
		Stdin:              stdin,
		Stdout:             stdout,
		Stderr:             stderr,
		Tty:                tty,
	})
}
