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

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/pod-security-admission/api"
	"k8s.io/pod-security-admission/test"
	"k8s.io/utils/ptr"

	"k8s.io/kubernetes/test/e2e/framework"
	e2edeployment "k8s.io/kubernetes/test/e2e/framework/deployment"
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
	user1creds                = "dXNlcjpwYXNzd29yZA==" // user:password
	registryNodeSelectorValue = "kube-e2e-test-registry"
)

func SetupRegistry(ctx context.Context, f *framework.Framework, podOnly bool) ([]string, error) {
	const registryReplicas = 1

	podTestLabel := "test-registry-pod-" + f.UniqueName
	pod, err := podManifest(podTestLabel)
	if err != nil {
		return nil, err
	}

	if podOnly {
		podClient := e2epod.NewPodClient(f)
		pod = podClient.Create(ctx, pod)
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod))
	} else {
		labels := map[string]string{"kube-e2e": podTestLabel}
		deployment := e2edeployment.NewDeployment("", registryReplicas, labels, "", "", appsv1.RollingUpdateDeploymentStrategyType)
		deployment.GenerateName = "test-registry-"
		deployment.Spec.Template.Spec = pod.Spec

		deployment, err = f.ClientSet.AppsV1().Deployments(f.Namespace.Name).Create(ctx, deployment, metav1.CreateOptions{})
		if err != nil {
			return nil, err
		}

		err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 120*time.Second, true, func(ctx context.Context) (done bool, err error) {
			dep, err := f.ClientSet.AppsV1().Deployments(f.Namespace.Name).Get(ctx, deployment.Name, metav1.GetOptions{})
			if err != nil {
				return false, nil
			}
			return dep.Status.UpdatedReplicas == registryReplicas &&
				dep.Status.AvailableReplicas == registryReplicas &&
				dep.Status.ReadyReplicas == registryReplicas, nil
		})
		if err != nil {
			return nil, err
		}
	}

	pods, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).List(ctx, metav1.ListOptions{LabelSelector: "kube-e2e=" + podTestLabel})
	if err != nil {
		return nil, err
	}
	podNodes := make([]string, 0, len(pods.Items))
	for _, pod := range pods.Items {
		podNodes = append(podNodes, pod.Spec.NodeName)
	}

	return podNodes, nil
}

func podManifest(podTestLabel string) (*v1.Pod, error) {
	pod, err := test.GetMinimalValidLinuxPod(api.LevelRestricted, api.MajorMinorVersion(1, 22))
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

	return pod, nil
}

func SetupRegistryLabelNodes(ctx context.Context, f *framework.Framework, podOnly bool) (cleanup func(context.Context) error, err error) {
	cleanups := []func(ctx context.Context) error{}
	cleanup = func(ctx context.Context) error {
		var errs []error
		for _, c := range cleanups {
			if cleanupError := c(ctx); cleanupError != nil {
				errs = append(errs, cleanupError)
			}
		}
		return utilerrors.NewAggregate(errs)
	}

	defer func() {
		if err != nil {
			if cleanupError := cleanup(ctx); cleanupError != nil {
				framework.Logf("cleanup failed: %v", err)
			}
		}
	}()

	var registryNodes []string
	registryNodes, err = SetupRegistry(ctx, f, podOnly)
	if err != nil {
		return cleanup, err
	}

	for _, nodeName := range registryNodes {
		if _, err := f.ClientSet.CoreV1().Nodes().Patch(ctx, nodeName, types.ApplyPatchType,
			[]byte(`{"kind":"Node","apiVersion":"v1","metadata":{"labels":{"`+f.UniqueName+`":"`+registryNodeSelectorValue+`"}}}`),
			metav1.PatchOptions{
				FieldManager: "e2e-tests",
			},
		); err != nil {
			return cleanup, err
		}
		cleanups = append(cleanups, func(ctx context.Context) error {
			node, nodeErr := f.ClientSet.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
			if nodeErr != nil {
				return nodeErr
			}
			delete(node.Labels, f.UniqueName)
			_, nodeErr = f.ClientSet.CoreV1().Nodes().Update(ctx, node, metav1.UpdateOptions{})
			return nodeErr
		})
	}

	return cleanup, err
}

func PodNodeSelector(f *framework.Framework) map[string]string {
	return map[string]string{f.UniqueName: registryNodeSelectorValue}
}

func User1DockerSecret(registryAddress string) *v1.Secret {
	return &v1.Secret{
		Type: v1.SecretTypeDockerConfigJson,
		Data: map[string][]byte{
			v1.DockerConfigJsonKey: fmt.Appendf(nil, dockerCredsFmt, "http://"+registryAddress, user1creds),
		},
	}
}
