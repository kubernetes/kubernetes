/*
Copyright 2024 The Kubernetes Authors.

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
	"bytes"
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	coreclientset "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/remotecommand"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubernetes/test/e2e/framework"
)

// ExecInPodWithResult runs a command in a container and returns the stdout.
func ExecInPodWithResult(ctx context.Context, podClient coreclientset.CoreV1Interface, podRESTConfig *rest.Config, ns, name, containerName string, command []string) (string, error) {
	u := podClient.RESTClient().Post().Resource("pods").Namespace(ns).Name(name).SubResource("exec").VersionedParams(&corev1.PodExecOptions{
		Container: containerName,
		Stdout:    true,
		Stderr:    true,
		Command:   command,
	}, scheme.ParameterCodec).URL()

	e, err := remotecommand.NewSPDYExecutor(podRESTConfig, "POST", u)
	if err != nil {
		return "", fmt.Errorf("could not initialize a new SPDY executor: %w", err)
	}
	buf := &bytes.Buffer{}
	errBuf := &bytes.Buffer{}
	if err := e.StreamWithContext(ctx,
		remotecommand.StreamOptions{
			Stdout: buf,
			Stdin:  nil,
			Stderr: errBuf,
		}); err != nil {
		framework.Logf("exec error: %s", errBuf.String())
		return "", err
	}
	return buf.String(), nil
}
