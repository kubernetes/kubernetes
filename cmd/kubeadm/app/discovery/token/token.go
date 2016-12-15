/*
Copyright 2016 The Kubernetes Authors.

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

package token

import (
	"net/url"
	"strings"

	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func Parse(u *url.URL, c *kubeadm.Discovery) error {
	var (
		hosts          []string
		tokenID, token string
	)
	if u.Host != "" {
		hosts = strings.Split(u.Host, ",")
	}
	if u.User != nil {
		if p, ok := u.User.Password(); ok {
			tokenID = u.User.Username()
			token = p
		}
	}
	c.Token = &kubeadm.TokenDiscovery{
		ID:        tokenID,
		Secret:    token,
		Addresses: hosts,
	}
	return nil
}
