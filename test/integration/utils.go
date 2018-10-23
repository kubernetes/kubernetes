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
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	clientset "k8s.io/client-go/kubernetes"
	coreclient "k8s.io/client-go/kubernetes/typed/core/v1"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/pkg/transport"
)

func DeletePodOrErrorf(t *testing.T, c clientset.Interface, ns, name string) {
	if err := c.CoreV1().Pods(ns).Delete(name, nil); err != nil {
		t.Errorf("unable to delete pod %v: %v", name, err)
	}
}

// Requests to try.  Each one should be forbidden or not forbidden
// depending on the authentication and authorization setup of the master.
var Code200 = map[int]bool{200: true}
var Code201 = map[int]bool{201: true}
var Code400 = map[int]bool{400: true}
var Code401 = map[int]bool{401: true}
var Code403 = map[int]bool{403: true}
var Code404 = map[int]bool{404: true}
var Code405 = map[int]bool{405: true}
var Code409 = map[int]bool{409: true}
var Code422 = map[int]bool{422: true}
var Code500 = map[int]bool{500: true}
var Code503 = map[int]bool{503: true}

// WaitForPodToDisappear polls the API server if the pod has been deleted.
func WaitForPodToDisappear(podClient coreclient.PodInterface, podName string, interval, timeout time.Duration) error {
	return wait.PollImmediate(interval, timeout, func() (bool, error) {
		_, err := podClient.Get(podName, metav1.GetOptions{})
		if err == nil {
			return false, nil
		} else {
			if errors.IsNotFound(err) {
				return true, nil
			} else {
				return false, err
			}
		}
	})
}

func GetEtcdClients(config storagebackend.Config) (*clientv3.Client, clientv3.KV, error) {
	tlsInfo := transport.TLSInfo{
		CertFile: config.CertFile,
		KeyFile:  config.KeyFile,
		CAFile:   config.CAFile,
	}

	tlsConfig, err := tlsInfo.ClientConfig()
	if err != nil {
		return nil, nil, err
	}

	cfg := clientv3.Config{
		Endpoints: config.ServerList,
		TLS:       tlsConfig,
	}

	c, err := clientv3.New(cfg)
	if err != nil {
		return nil, nil, err
	}

	return c, clientv3.NewKV(c), nil
}
