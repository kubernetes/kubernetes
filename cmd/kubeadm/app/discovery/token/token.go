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
	"strings"

	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func Parse(s string, c *kubeadm.Discovery) error {
	c.Token = &kubeadm.TokenDiscovery{}

	// Make sure there's something to parse after discarding the scheme fragment, "token://"
	if s[8:] == "" {
		return nil
	}

	// Discard scheme and split token fragment from host fragment
	urlFragments := strings.Split(s[8:], "@")

	// Extract token from token fragment
	// TODO what if an invalid token is passed, e.g. token://xxxx?
	rawToken := strings.Split(urlFragments[0], ":")
	c.Token.ID = rawToken[0]
	c.Token.Secret = rawToken[1]

	// Extract hosts, if host fragment exists
	if len(urlFragments) == 2 {
		c.Token.Addresses = strings.Split(urlFragments[1], ",")
	}

	return nil
}
