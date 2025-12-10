/*
Copyright 2025 The Kubernetes Authors.

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

package registry

import (
	"context"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/pod-security-admission/api"
	"k8s.io/pod-security-admission/test"
	"k8s.io/utils/ptr"

	"k8s.io/kubernetes/test/e2e/framework"
	e2edaemonset "k8s.io/kubernetes/test/e2e/framework/daemonset"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	dockerCredsFmt = `{
	"auths": {
		"%s": {
			"auth": "%s"
		}
	}
}`
	user1creds = "dXNlcjpwYXNzd29yZA==" // user:password
)

// SetupRegistry runs the `fake-registry-server --private` from the agnhost image.
// The registry is run with HostPort 5000 exposed in order to allow locally-scheduled
// pods to query the registry via kubelet.
// The registry only runs in HTTP (no TLS) mode, and so this hack-path is used as
// localhost is typically allowed by CRIs and so no CRI-specific configuration is needed
//
// By default, the function runs the registry as a DaemonSet on all nodes, but it supports running
// it in just a `pod` for cases where kube-controller-manager is not running (like in
// the Node Conformance test suite).
//
// This function returns:
// - the node-local address of the registry
// - set of node names that the registry runs on, mostly useful only in the podOnly case
// - an error
//
// TODO: once https://github.com/kubernetes/kubernetes/issues/132955 is
// addressed, we might be able to proxy a single endpoint from the cluster to each
// node's localhost port instead of using DaemonSets.
func SetupRegistry(ctx context.Context, f *framework.Framework, podOnly bool) (string, []string, error) {
	podTestLabel := "test-registry-pod-" + f.UniqueName
	pod, err := podManifest(podTestLabel)
	if err != nil {
		return "", nil, err
	}

	if podOnly {
		podClient := e2epod.NewPodClient(f)
		pod = podClient.Create(ctx, pod)
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod))
	} else {
		labels := map[string]string{"kube-e2e": podTestLabel}
		daemonset := e2edaemonset.NewDaemonSet("", "", labels, nil, nil, nil)
		daemonset.GenerateName = "test-registry-"
		daemonset.Spec.Template.Spec = pod.Spec

		daemonset, err = f.ClientSet.AppsV1().DaemonSets(f.Namespace.Name).Create(ctx, daemonset, metav1.CreateOptions{})
		if err != nil {
			return "", nil, err
		}

		err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 120*time.Second, true, func(ctx context.Context) (done bool, err error) {
			return e2edaemonset.CheckRunningOnAllNodes(ctx, f, daemonset)
		})
		if err != nil {
			return "", nil, err
		}
	}

	pods, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).List(ctx, metav1.ListOptions{LabelSelector: "kube-e2e=" + podTestLabel})
	if err != nil {
		return "", nil, err
	}
	podNodes := make([]string, 0, len(pods.Items))
	for _, pod := range pods.Items {
		podNodes = append(podNodes, pod.Spec.NodeName)
	}

	return "localhost:5000", podNodes, nil
}

func podManifest(podTestLabel string) (*v1.Pod, error) {
	pod, err := test.GetMinimalValidPod(api.LevelRestricted, api.MajorMinorVersion(1, 25))
	if err != nil {
		return nil, err
	}

	pod.ObjectMeta.GenerateName = "test-registry-"
	pod.ObjectMeta.Labels = map[string]string{"kube-e2e": podTestLabel}

	pod.Spec.InitContainers = nil
	pod.Spec.Containers[0].Name = "registry"
	pod.Spec.Containers[0].Image = imageutils.GetE2EImage(imageutils.Agnhost)
	pod.Spec.Containers[0].ImagePullPolicy = v1.PullIfNotPresent
	pod.Spec.Containers[0].Args = []string{"fake-registry-server", "--private"}
	pod.Spec.Containers[0].Ports = []v1.ContainerPort{
		{
			Name:          "http",
			ContainerPort: 5000,
			HostPort:      5000,
		},
	}
	pod.Spec.Containers[0].SecurityContext.RunAsUser = ptr.To[int64](5123)
	// setting HostNetwork to true so that the registry is accessible on localhost:<hostport>
	// and we don't have to deal with any CNI quirks.
	pod.Spec.HostNetwork = true

	return pod, nil
}

// User1DockerSecret creates a secret containing the docker credentials for pulling from
// the agnhost fake-registry-server.
func User1DockerSecret(registryAddress string) *v1.Secret {
	return &v1.Secret{
		Type: v1.SecretTypeDockerConfigJson,
		Data: map[string][]byte{
			v1.DockerConfigJsonKey: fmt.Appendf(nil, dockerCredsFmt, registryAddress, user1creds),
		},
	}
}
