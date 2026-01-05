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

	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
	"k8s.io/utils/ptr"
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
	flag.Parse()

	config, err := clientcmd.BuildConfigFromFlags("", *kubeconfig)
	if err != nil {
		panic(err.Error())
	}
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		panic(err.Error())
	}

	pods, err := clientset.CoreV1().Pods("").List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		panic(err.Error())
	}

	for _, pod := range pods.Items {
		podName := pod.ObjectMeta.Name
		namespace := pod.ObjectMeta.Namespace
		podIP := pod.Status.PodIP
		nodeName := pod.Spec.NodeName

		fmt.Printf("Name: %s, Namespace: %s, IP: %s, Node: %s\n", podName, namespace, podIP, nodeName)

		for _, container := range pod.Spec.Containers {
			containerName := container.Name
			if err := getPodLogs(clientset, namespace, podName, containerName); err != nil {
				fmt.Println(err)
			}
		}
	}
}

func getPodLogs(clientset *kubernetes.Clientset, namespace, podName, containerName string) (err error) {
	fmt.Printf("Logs for container %s:\n", containerName)

	podLogOpts := apiv1.PodLogOptions{
		Container: containerName,
		TailLines: ptr.To(int64(5)),
	}

	req := clientset.CoreV1().Pods(namespace).GetLogs(podName, &podLogOpts)
	var podLogs io.ReadCloser
	podLogs, err = req.Stream(context.TODO())
	if err != nil {
		return fmt.Errorf("error getting logs for pod %s, container %s: %w", podName, containerName, err)
	}
	defer func() {
		cerr := podLogs.Close()
		if err == nil {
			err = cerr
		}
	}()

	_, err = io.Copy(os.Stdout, podLogs)
	if err != nil {
		return fmt.Errorf("error copying logs for pod %s, container %s: %w", podName, containerName, err)
	}
	fmt.Println()
	return nil
}
