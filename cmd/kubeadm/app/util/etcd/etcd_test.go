/*
Copyright 2018 The Kubernetes Authors.

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

package etcd

import (
	"fmt"
	"reflect"
	"strconv"
	"testing"

	"github.com/pkg/errors"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	testresources "k8s.io/kubernetes/cmd/kubeadm/test/resources"
)

func testGetURL(t *testing.T, getURLFunc func(*kubeadmapi.APIEndpoint) string, port int) {
	portStr := strconv.Itoa(port)
	var tests = []struct {
		name             string
		advertiseAddress string
		expectedURL      string
	}{
		{
			name:             "IPv4",
			advertiseAddress: "10.10.10.10",
			expectedURL:      fmt.Sprintf("https://10.10.10.10:%s", portStr),
		},
		{
			name:             "IPv6",
			advertiseAddress: "2001:db8::2",
			expectedURL:      fmt.Sprintf("https://[2001:db8::2]:%s", portStr),
		},
		{
			name:             "IPv4 localhost",
			advertiseAddress: "127.0.0.1",
			expectedURL:      fmt.Sprintf("https://127.0.0.1:%s", portStr),
		},
		{
			name:             "IPv6 localhost",
			advertiseAddress: "::1",
			expectedURL:      fmt.Sprintf("https://[::1]:%s", portStr),
		},
	}

	for _, test := range tests {
		url := getURLFunc(&kubeadmapi.APIEndpoint{AdvertiseAddress: test.advertiseAddress})
		if url != test.expectedURL {
			t.Errorf("expected %s, got %s", test.expectedURL, url)
		}
	}
}

func TestGetClientURL(t *testing.T) {
	testGetURL(t, GetClientURL, constants.EtcdListenClientPort)
}

func TestGetPeerURL(t *testing.T) {
	testGetURL(t, GetClientURL, constants.EtcdListenClientPort)
}

func TestGetClientURLByIP(t *testing.T) {
	portStr := strconv.Itoa(constants.EtcdListenClientPort)
	var tests = []struct {
		name        string
		ip          string
		expectedURL string
	}{
		{
			name:        "IPv4",
			ip:          "10.10.10.10",
			expectedURL: fmt.Sprintf("https://10.10.10.10:%s", portStr),
		},
		{
			name:        "IPv6",
			ip:          "2001:db8::2",
			expectedURL: fmt.Sprintf("https://[2001:db8::2]:%s", portStr),
		},
		{
			name:        "IPv4 localhost",
			ip:          "127.0.0.1",
			expectedURL: fmt.Sprintf("https://127.0.0.1:%s", portStr),
		},
		{
			name:        "IPv6 localhost",
			ip:          "::1",
			expectedURL: fmt.Sprintf("https://[::1]:%s", portStr),
		},
	}

	for _, test := range tests {
		url := GetClientURLByIP(test.ip)
		if url != test.expectedURL {
			t.Errorf("expected %s, got %s", test.expectedURL, url)
		}
	}
}

func TestGetEtcdEndpointsWithBackoff(t *testing.T) {
	var tests = []struct {
		name              string
		pods              []testresources.FakeStaticPod
		configMap         *testresources.FakeConfigMap
		expectedEndpoints []string
		expectedErr       bool
	}{
		{
			name:              "no pod annotations; no ClusterStatus",
			expectedEndpoints: []string{},
		},
		{
			name: "ipv4 endpoint in pod annotation; no ClusterStatus; port is preserved",
			pods: []testresources.FakeStaticPod{
				{
					Component: constants.Etcd,
					Annotations: map[string]string{
						constants.EtcdAdvertiseClientUrlsAnnotationKey: "https://1.2.3.4:1234",
					},
				},
			},
			expectedEndpoints: []string{"https://1.2.3.4:1234"},
		},
		{
			name:              "no pod annotations; ClusterStatus with valid ipv4 endpoint; port is inferred",
			configMap:         testresources.ClusterStatusWithAPIEndpoint("cp-0", kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4", BindPort: 1234}),
			expectedEndpoints: []string{"https://1.2.3.4:2379"},
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			for _, pod := range rt.pods {
				if err := pod.Create(client); err != nil {
					t.Errorf("error setting up test creating pod for node %q", pod.NodeName)
				}
			}
			if rt.configMap != nil {
				if err := rt.configMap.Create(client); err != nil {
					t.Error("could not create ConfigMap")
				}
			}
			endpoints, err := getEtcdEndpointsWithBackoff(client, wait.Backoff{Duration: 0, Jitter: 0, Steps: 1})
			if err != nil && !rt.expectedErr {
				t.Errorf("got error %q; was expecting no errors", err)
				return
			} else if err == nil && rt.expectedErr {
				t.Error("got no error; was expecting an error")
				return
			} else if err != nil && rt.expectedErr {
				return
			}

			if !reflect.DeepEqual(endpoints, rt.expectedEndpoints) {
				t.Errorf("expected etcd endpoints: %v; got: %v", rt.expectedEndpoints, endpoints)
			}
		})
	}
}

func TestGetRawEtcdEndpointsFromPodAnnotation(t *testing.T) {
	var tests = []struct {
		name              string
		pods              []testresources.FakeStaticPod
		clientSetup       func(*clientsetfake.Clientset)
		expectedEndpoints []string
		expectedErr       bool
	}{
		{
			name: "exactly one pod with annotation",
			pods: []testresources.FakeStaticPod{
				{
					NodeName:    "cp-0",
					Component:   constants.Etcd,
					Annotations: map[string]string{constants.EtcdAdvertiseClientUrlsAnnotationKey: "https://1.2.3.4:2379"},
				},
			},
			expectedEndpoints: []string{"https://1.2.3.4:2379"},
		},
		{
			name:        "no pods with annotation",
			expectedErr: true,
		},
		{
			name: "exactly one pod with annotation; all requests fail",
			pods: []testresources.FakeStaticPod{
				{
					NodeName:    "cp-0",
					Component:   constants.Etcd,
					Annotations: map[string]string{constants.EtcdAdvertiseClientUrlsAnnotationKey: "https://1.2.3.4:2379"},
				},
			},
			clientSetup: func(clientset *clientsetfake.Clientset) {
				clientset.PrependReactor("list", "pods", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, nil, apierrors.NewInternalError(errors.New("API server down"))
				})
			},
			expectedErr: true,
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			for i, pod := range rt.pods {
				if err := pod.CreateWithPodSuffix(client, strconv.Itoa(i)); err != nil {
					t.Errorf("error setting up test creating pod for node %q", pod.NodeName)
				}
			}
			if rt.clientSetup != nil {
				rt.clientSetup(client)
			}
			endpoints, err := getRawEtcdEndpointsFromPodAnnotation(client, wait.Backoff{Duration: 0, Jitter: 0, Steps: 1})
			if err != nil && !rt.expectedErr {
				t.Errorf("got error %v, but wasn't expecting any error", err)
				return
			} else if err == nil && rt.expectedErr {
				t.Error("didn't get any error; but was expecting an error")
				return
			} else if err != nil && rt.expectedErr {
				return
			}
			if !reflect.DeepEqual(endpoints, rt.expectedEndpoints) {
				t.Errorf("expected etcd endpoints: %v; got: %v", rt.expectedEndpoints, endpoints)
			}
		})
	}
}

func TestGetRawEtcdEndpointsFromPodAnnotationWithoutRetry(t *testing.T) {
	var tests = []struct {
		name              string
		pods              []testresources.FakeStaticPod
		clientSetup       func(*clientsetfake.Clientset)
		expectedEndpoints []string
		expectedErr       bool
	}{
		{
			name:              "no pods",
			expectedEndpoints: []string{},
		},
		{
			name: "exactly one pod with annotation",
			pods: []testresources.FakeStaticPod{
				{
					NodeName:    "cp-0",
					Component:   constants.Etcd,
					Annotations: map[string]string{constants.EtcdAdvertiseClientUrlsAnnotationKey: "https://1.2.3.4:2379"},
				},
			},
			expectedEndpoints: []string{"https://1.2.3.4:2379"},
		},
		{
			name: "two pods with annotation",
			pods: []testresources.FakeStaticPod{
				{
					NodeName:    "cp-0",
					Component:   constants.Etcd,
					Annotations: map[string]string{constants.EtcdAdvertiseClientUrlsAnnotationKey: "https://1.2.3.4:2379"},
				},
				{
					NodeName:    "cp-1",
					Component:   constants.Etcd,
					Annotations: map[string]string{constants.EtcdAdvertiseClientUrlsAnnotationKey: "https://1.2.3.5:2379"},
				},
			},
			expectedEndpoints: []string{"https://1.2.3.4:2379", "https://1.2.3.5:2379"},
		},
		{
			name: "exactly one pod with annotation; request fails",
			pods: []testresources.FakeStaticPod{
				{
					NodeName:    "cp-0",
					Component:   constants.Etcd,
					Annotations: map[string]string{constants.EtcdAdvertiseClientUrlsAnnotationKey: "https://1.2.3.4:2379"},
				},
			},
			clientSetup: func(clientset *clientsetfake.Clientset) {
				clientset.PrependReactor("list", "pods", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, nil, apierrors.NewInternalError(errors.New("API server down"))
				})
			},
			expectedErr: true,
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			for _, pod := range rt.pods {
				if err := pod.Create(client); err != nil {
					t.Errorf("error setting up test creating pod for node %q", pod.NodeName)
					return
				}
			}
			if rt.clientSetup != nil {
				rt.clientSetup(client)
			}
			endpoints, _, err := getRawEtcdEndpointsFromPodAnnotationWithoutRetry(client)
			if err != nil && !rt.expectedErr {
				t.Errorf("got error %v, but wasn't expecting any error", err)
				return
			} else if err == nil && rt.expectedErr {
				t.Error("didn't get any error; but was expecting an error")
				return
			} else if err != nil && rt.expectedErr {
				return
			}
			if !reflect.DeepEqual(endpoints, rt.expectedEndpoints) {
				t.Errorf("expected etcd endpoints: %v; got: %v", rt.expectedEndpoints, endpoints)
			}
		})
	}
}
