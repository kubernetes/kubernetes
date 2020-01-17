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
	"strconv"
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
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
