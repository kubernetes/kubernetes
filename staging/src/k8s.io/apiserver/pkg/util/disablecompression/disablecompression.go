/*
Copyright 2022 The Kubernetes Authors.

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

package disablecompression

import (
	"fmt"
	"net"
	"net/http"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	netutils "k8s.io/utils/net"
)

// ClientIPPredicate.Predicate implements CompressionDisabledFunc interface that decides
// based on client IP.
type ClientIPPredicate struct {
	cidrs []*net.IPNet
}

// NewClientIPPredicate creates a new ClientIPPredicate instance.
func NewClientIPPredicate(cidrStrings []string) (*ClientIPPredicate, error) {
	cidrs, err := netutils.ParseCIDRs(cidrStrings)
	if err != nil {
		return nil, fmt.Errorf("failed to parse cidrs: %v", err)
	}
	return &ClientIPPredicate{cidrs: cidrs}, nil
}

// Predicate checks if ClientIP matches any cidr.
func (c *ClientIPPredicate) Predicate(req *http.Request) (bool, error) {
	ip := utilnet.GetClientIP(req)
	if ip == nil {
		return false, fmt.Errorf("unable to determine source IP for %v", req)
	}

	for _, cidr := range c.cidrs {
		if cidr.Contains(ip) {
			return true, nil
		}
	}

	return false, nil
}
