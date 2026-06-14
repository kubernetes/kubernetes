/*
Copyright The Kubernetes Authors.

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

// Note: the example only works with the code within the same release/branch.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
	//
	// Uncomment to load all auth plugins
	// _ "k8s.io/client-go/plugin/pkg/client/auth"
	//
	// Or uncomment to load specific auth plugins
	// _ "k8s.io/client-go/plugin/pkg/client/auth/azure"
	// _ "k8s.io/client-go/plugin/pkg/client/auth/gcp"
	// _ "k8s.io/client-go/plugin/pkg/client/auth/oidc"
)

func main() {
	var kubeconfig *string
	if home := homedir.HomeDir(); home != "" {
		kubeconfig = flag.String("kubeconfig", filepath.Join(home, ".kube", "config"), "(optional) absolute path to the kubeconfig file")
	} else {
		kubeconfig = flag.String("kubeconfig", "", "absolute path to the kubeconfig file")
	}

	namespace := flag.String("namespace", metav1.NamespaceAll, "namespace to read pod logs from; empty means all namespaces")
	podName := flag.String("pod", "", "optional pod name to read logs from")
	containerName := flag.String("container", "", "optional container name to read logs from")
	tailLines := flag.Int64("tail", 10, "number of log lines to read from each container")
	follow := flag.Bool("follow", false, "stream logs as new log lines are written")
	flag.Parse()

	config, err := clientcmd.BuildConfigFromFlags("", *kubeconfig)
	if err != nil {
		panic(err)
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		panic(err)
	}

	ctx := context.Background()
	pods, err := clientset.CoreV1().Pods(*namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		panic(err)
	}

	for _, pod := range pods.Items {
		if *podName != "" && pod.Name != *podName {
			continue
		}
		for _, container := range containersForPod(pod, *containerName) {
			if err := streamLogs(ctx, clientset, pod.Namespace, pod.Name, container, *tailLines, *follow); err != nil {
				fmt.Fprintf(os.Stderr, "error reading logs for %s/%s/%s: %v\n", pod.Namespace, pod.Name, container, err)
			}
		}
	}
}

func containersForPod(pod corev1.Pod, containerName string) []string {
	if containerName != "" {
		return []string{containerName}
	}

	containers := make([]string, 0, len(pod.Spec.InitContainers)+len(pod.Spec.Containers)+len(pod.Spec.EphemeralContainers))
	for _, container := range pod.Spec.InitContainers {
		containers = append(containers, container.Name)
	}
	for _, container := range pod.Spec.Containers {
		containers = append(containers, container.Name)
	}
	for _, container := range pod.Spec.EphemeralContainers {
		containers = append(containers, container.Name)
	}
	return containers
}

func streamLogs(ctx context.Context, clientset *kubernetes.Clientset, namespace, podName, containerName string, tailLines int64, follow bool) error {
	fmt.Printf("==> %s/%s/%s <==\n", namespace, podName, containerName)

	req := clientset.CoreV1().Pods(namespace).GetLogs(podName, &corev1.PodLogOptions{
		Container: containerName,
		Follow:    follow,
		TailLines: &tailLines,
	})
	stream, err := req.Stream(ctx)
	if err != nil {
		return err
	}
	defer stream.Close()

	_, err = io.Copy(os.Stdout, stream)
	if err != nil {
		return err
	}
	fmt.Println()
	return nil
}
