/*
Copyright 2014 The Kubernetes Authors.

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

package integration

import (
	"context"
	"testing"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	clientset "k8s.io/client-go/kubernetes"
	coreclient "k8s.io/client-go/kubernetes/typed/core/v1"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
)

// DeletePodOrErrorf deletes a pod or fails with a call to t.Errorf.
func DeletePodOrErrorf(t *testing.T, c clientset.Interface, ns, name string) {
	if err := c.CoreV1().Pods(ns).Delete(context.TODO(), name, metav1.DeleteOptions{}); err != nil {
		t.Errorf("unable to delete pod %v: %v", name, err)
	}
}

// Requests to try.  Each one should be forbidden or not forbidden
// depending on the authentication and authorization setup of the API server.
var (
	Code200 = map[int]bool{200: true}
	Code201 = map[int]bool{201: true}
	Code400 = map[int]bool{400: true}
	Code401 = map[int]bool{401: true}
	Code403 = map[int]bool{403: true}
	Code404 = map[int]bool{404: true}
	Code405 = map[int]bool{405: true}
	Code503 = map[int]bool{503: true}
)

// WaitForPodToDisappear polls the API server if the pod has been deleted.
func WaitForPodToDisappear(podClient coreclient.PodInterface, podName string, interval, timeout time.Duration) error {
	return wait.PollImmediate(interval, timeout, func() (bool, error) {
		_, err := podClient.Get(context.TODO(), podName, metav1.GetOptions{})
		if err == nil {
			return false, nil
		}
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	})
}

// GetEtcdClients returns an initialized etcd clientv3.Client and clientv3.KV.
func GetEtcdClients(config storagebackend.TransportConfig) (*clientv3.Client, clientv3.KV, error) {
	return kubeapiservertesting.GetEtcdClients(config)
}
