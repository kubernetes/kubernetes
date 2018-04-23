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

package apiclient

import (
	"fmt"
	"io"
	"net/http"
	"time"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	netutil "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/kubeletclient"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

// Waiter is an interface for waiting for criteria in Kubernetes to happen
type Waiter interface {
	// WaitForAPI waits for the API Server's /healthz endpoint to become "ok"
	WaitForAPI() error
	WaitForEtcd(secure bool) error
	// WaitForPodsWithLabel waits for Pods in the kube-system namespace to become Ready
	WaitForPodsWithLabel(kvLabel string) error
	// WaitForPodToDisappear waits for the given Pod in the kube-system namespace to be deleted
	WaitForPodToDisappear(staticPodName string) error
	// WaitForStaticPodHash fetches static pod hash for the requested component
	WaitForStaticPodHash(nodeName string, component string) (string, error)
	// WaitForStaticPodHashChange waits for the given component's static pod hash to be updated.
	// By doing that we can be sure that the kubelet has picked up the new static pod manifest.
	WaitForStaticPodHashChange(nodeName, component, previousHash string) error
	// WaitForMirrorPodHashChange waits for the given component's static pod hash to be updated.
	// By doing that we can be sure that the kubelet has restarted the given Static Pod
	WaitForMirrorPodHashChange(nodeName, component, previousHash string) error
	// WaitForStaticPodControlPlaneHashes fetches sha256 hashes for the control plane static pods
	WaitForStaticPodControlPlaneHashes(nodeName string) (map[string]string, error)
	// WaitForHealthyKubelet blocks until the kubelet /healthz endpoint returns 'ok'
	WaitForHealthyKubelet(initalTimeout time.Duration, healthzEndpoint string) error
	// SetTimeout adjusts the timeout to the specified duration
	SetTimeout(timeout time.Duration)
}

// KubeWaiter is an implementation of Waiter that is backed by a Kubernetes client
type KubeWaiter struct {
	client        clientset.Interface
	kubeletClient kubeletclient.KubeletClient
	timeout       time.Duration
	writer        io.Writer
}

// NewKubeWaiter returns a new Waiter object that talks to the given Kubernetes cluster
func NewKubeWaiter(client clientset.Interface, kubeletClient kubeletclient.KubeletClient, timeout time.Duration, writer io.Writer) Waiter {
	return &KubeWaiter{
		client:        client,
		kubeletClient: kubeletClient,
		timeout:       timeout,
		writer:        writer,
	}
}

// WaitForAPI waits for the API Server's /healthz endpoint to report "ok"
func (w *KubeWaiter) WaitForAPI() error {
	start := time.Now()
	return wait.PollImmediate(constants.APICallRetryInterval, w.timeout, func() (bool, error) {
		healthStatus := 0
		w.client.Discovery().RESTClient().Get().AbsPath("/healthz").Do().StatusCode(&healthStatus)
		if healthStatus != http.StatusOK {
			return false, nil
		}

		fmt.Printf("[apiclient] All control plane components are healthy after %f seconds\n", time.Since(start).Seconds())
		return true, nil
	})
}

// WaitForPodsWithLabel will lookup pods with the given label and wait until they are all
// reporting status as running.
func (w *KubeWaiter) WaitForPodsWithLabel(kvLabel string) error {

	lastKnownPodNumber := -1
	return wait.PollImmediate(constants.APICallRetryInterval, w.timeout, func() (bool, error) {
		listOpts := metav1.ListOptions{LabelSelector: kvLabel}
		pods, err := w.client.CoreV1().Pods(metav1.NamespaceSystem).List(listOpts)
		if err != nil {
			fmt.Fprintf(w.writer, "[apiclient] Error getting Pods with label selector %q [%v]\n", kvLabel, err)
			return false, nil
		}

		if lastKnownPodNumber != len(pods.Items) {
			fmt.Fprintf(w.writer, "[apiclient] Found %d Pods for label selector %s\n", len(pods.Items), kvLabel)
			lastKnownPodNumber = len(pods.Items)
		}

		if len(pods.Items) == 0 {
			return false, nil
		}

		for _, pod := range pods.Items {
			if pod.Status.Phase != v1.PodRunning {
				return false, nil
			}
		}

		return true, nil
	})
}

// WaitForPodToDisappear blocks until it timeouts or gets a "NotFound" response from the API Server when getting the Static Pod in question
func (w *KubeWaiter) WaitForPodToDisappear(podName string) error {
	return wait.PollImmediate(constants.APICallRetryInterval, w.timeout, func() (bool, error) {
		_, err := w.client.CoreV1().Pods(metav1.NamespaceSystem).Get(podName, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			fmt.Printf("[apiclient] The old Pod %q is now removed (which is desired)\n", podName)
			return true, nil
		}
		return false, nil
	})
}

// WaitForEtcd blocks until it timeouts or gets a status response from etcd
func (w *KubeWaiter) WaitForEtcd(secure bool) error {
	return TryRunCommand(func() error {
		etcdCluster := kubeadmutil.LocalEtcdCluster{}
		if secure {
			_, err := etcdCluster.GetSecureEtcdClusterStatus()
			if err != nil {
				fmt.Printf("[etcd-check] It seems like etcd isn't running or healthy.\n")
				fmt.Printf("[etcd-check] failed with error: %v.\n", err)
				return err
			}
		} else {
			_, err := etcdCluster.GetEtcdClusterStatus()
			if err != nil {
				fmt.Printf("[etcd-check] It seems like etcd isn't running or healthy.\n")
				fmt.Printf("[etcd-check] failed with error: %v.\n", err)
				return err
			}
		}
		return nil
	}, 5)
}

// WaitForHealthyKubelet blocks until the kubelet /healthz endpoint returns 'ok'
func (w *KubeWaiter) WaitForHealthyKubelet(initalTimeout time.Duration, healthzEndpoint string) error {
	time.Sleep(initalTimeout)
	return TryRunCommand(func() error {
		client := &http.Client{Transport: netutil.SetOldTransportDefaults(&http.Transport{})}
		resp, err := client.Get(healthzEndpoint)
		if err != nil {
			fmt.Printf("[kubelet-check] It seems like the kubelet isn't running or healthy.\n")
			fmt.Printf("[kubelet-check] The HTTP call equal to 'curl -sSL %s' failed with error: %v.\n", healthzEndpoint, err)
			return err
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			fmt.Printf("[kubelet-check] It seems like the kubelet isn't running or healthy.")
			fmt.Printf("[kubelet-check] The HTTP call equal to 'curl -sSL %s' returned HTTP code %d\n", healthzEndpoint, resp.StatusCode)
			return fmt.Errorf("the kubelet healthz endpoint is unhealthy")
		}
		return nil
	}, 5) // a failureThreshold of five means waiting for a total of 155 seconds
}

// SetTimeout adjusts the timeout to the specified duration
func (w *KubeWaiter) SetTimeout(timeout time.Duration) {
	w.timeout = timeout
}

// WaitForStaticPodControlPlaneHashes blocks until it timeouts or gets a hash map for all components and their Static Pods
func (w *KubeWaiter) WaitForStaticPodControlPlaneHashes(nodeName string) (map[string]string, error) {

	componentHash := ""
	var err error
	mirrorPodHashes := map[string]string{}
	for _, component := range constants.MasterComponents {
		err = wait.PollImmediate(constants.APICallRetryInterval, w.timeout, func() (bool, error) {
			componentHash, err = getStaticPodHash(w.kubeletClient, nodeName, component)
			if err != nil {
				return false, nil
			}
			return true, nil
		})
		if err != nil {
			return nil, err
		}
		mirrorPodHashes[component] = componentHash
	}

	return mirrorPodHashes, nil
}

// WaitForStaticPodHash blocks until it timeouts or gets a hash for a single component and its Static Pod
func (w *KubeWaiter) WaitForStaticPodHash(nodeName string, component string) (string, error) {

	componentPodHash := ""
	var err error
	err = wait.PollImmediate(constants.APICallRetryInterval, w.timeout, func() (bool, error) {
		componentPodHash, err = getStaticPodHash(w.kubeletClient, nodeName, component)
		if err != nil {
			return false, nil
		}
		return true, nil
	})

	return componentPodHash, err
}

// WaitForMirrorPodHashChange blocks until it timeouts or notices that the Mirror Pod (for the Static Pod, respectively) has changed
// This implicitly means this function blocks until the kubelet has restarted the Static Pod in question
func (w *KubeWaiter) WaitForMirrorPodHashChange(nodeName, component, previousHash string) error {
	return wait.PollImmediate(constants.APICallRetryInterval, w.timeout, func() (bool, error) {

		hash, err := getMirrorPodHash(w.client, nodeName, component)
		if err != nil {
			return false, nil
		}
		// We should continue polling until the UID changes
		if hash == previousHash {
			return false, nil
		}

		return true, nil
	})
}

// WaitForStaticPodHashChange blocks until it timeouts or notices that the Mirror Pod (for the Static Pod, respectively) has changed
// This implicitly means this function blocks until the kubelet has restarted the Static Pod in question
func (w *KubeWaiter) WaitForStaticPodHashChange(nodeName, component, previousHash string) error {
	return wait.PollImmediate(constants.APICallRetryInterval, w.timeout, func() (bool, error) {

		hash, err := getStaticPodHash(w.kubeletClient, nodeName, component)
		if err != nil {
			return false, nil
		}
		// We should continue polling until the UID changes
		if hash == previousHash {
			return false, nil
		}

		return true, nil
	})
}

// getMirrorPodHash retrieves the static pod hash from the api server
func getMirrorPodHash(client clientset.Interface, nodeName string, component string) (string, error) {
	mirrorPodName := fmt.Sprintf("%s-%s", component, nodeName)
	mirrorPod, err := client.CoreV1().Pods(metav1.NamespaceSystem).Get(mirrorPodName, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	mirrorPodHash := mirrorPod.Annotations[kubetypes.ConfigHashAnnotationKey]
	return mirrorPodHash, nil
}

// getStaticPodHash retrieves the static pod hash from the kubelet api
func getStaticPodHash(client kubeletclient.KubeletClient, nodeName string, component string) (string, error) {
	staticPodName := fmt.Sprintf("%s-%s", component, nodeName)
	return client.GetStaticPodConfigHash(staticPodName)
}

// TryRunCommand runs a function a maximum of failureThreshold times, and retries on error. If failureThreshold is hit; the last error is returned
func TryRunCommand(f func() error, failureThreshold int) error {
	backoff := wait.Backoff{
		Duration: 5 * time.Second,
		Factor:   2, // double the timeout for every failure
		Steps:    failureThreshold,
	}
	return wait.ExponentialBackoff(backoff, func() (bool, error) {
		err := f()
		if err != nil {
			// Retry until the timeout
			return false, nil
		}
		// The last f() call was a success, return cleanly
		return true, nil
	})
}
