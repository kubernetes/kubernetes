/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package app

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api"
)

type fakeNodeInterface struct {
	node api.Node
}

func (fake *fakeNodeInterface) Get(hostname string) (*api.Node, error) {
	return &fake.node, nil
}

func Test_mayTryIptablesProxy(t *testing.T) {
	var cases = []struct {
		flag     string
		annKey   string
		annVal   string
		expected bool
	}{
		{"userspace", "", "", false},
		{"iptables", "", "", true},
		{"", "", "", false},
		{"", "net.experimental.kubernetes.io/proxy-mode", "userspace", false},
		{"", "net.experimental.kubernetes.io/proxy-mode", "iptables", true},
		{"", "net.experimental.kubernetes.io/proxy-mode", "other", false},
		{"", "net.experimental.kubernetes.io/proxy-mode", "", false},
		{"", "net.beta.kubernetes.io/proxy-mode", "userspace", false},
		{"", "net.beta.kubernetes.io/proxy-mode", "iptables", true},
		{"", "net.beta.kubernetes.io/proxy-mode", "other", false},
		{"", "net.beta.kubernetes.io/proxy-mode", "", false},
		{"", "proxy-mode", "iptables", false},
		{"userspace", "net.experimental.kubernetes.io/proxy-mode", "userspace", false},
		{"userspace", "net.experimental.kubernetes.io/proxy-mode", "iptables", false},
		{"iptables", "net.experimental.kubernetes.io/proxy-mode", "userspace", true},
		{"iptables", "net.experimental.kubernetes.io/proxy-mode", "iptables", true},
		{"userspace", "net.beta.kubernetes.io/proxy-mode", "userspace", false},
		{"userspace", "net.beta.kubernetes.io/proxy-mode", "iptables", false},
		{"iptables", "net.beta.kubernetes.io/proxy-mode", "userspace", true},
		{"iptables", "net.beta.kubernetes.io/proxy-mode", "iptables", true},
	}
	for i, c := range cases {
		getter := &fakeNodeInterface{}
		getter.node.Annotations = map[string]string{c.annKey: c.annVal}
		r := mayTryIptablesProxy(c.flag, getter, "host")
		if r != c.expected {
			t.Errorf("Case[%d] Expected %t, got %t", i, c.expected, r)
		}
	}
}

//This test verifies that Proxy Server does not crash that means
//Config and iptinterface are not nil when CleanupAndExit is true.
//To avoid proxy crash: https://github.com/kubernetes/kubernetes/pull/14736
func TestProxyServerWithCleanupAndExit(t *testing.T) {

	//creates default config
	config := NewProxyConfig()

	//sets CleanupAndExit manually
	config.CleanupAndExit = true

	//creates new proxy server
	proxyserver, err := NewProxyServerDefault(config)

	//verifies that nothing is nill except error
	assert.Nil(t, err)
	assert.NotNil(t, proxyserver)
	assert.NotNil(t, proxyserver.Config)
	assert.NotNil(t, proxyserver.IptInterface)
}
