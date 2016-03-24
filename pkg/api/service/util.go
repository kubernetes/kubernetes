/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package service

import (
	"fmt"
	"strings"

	netsets "k8s.io/kubernetes/pkg/util/net/sets"
)

const (
	defaultLoadBalancerSourceRanges = "0.0.0.0/0"
)

// IsAllowAll checks whether the netsets.IPNet allows traffic from 0.0.0.0/0
func IsAllowAll(ipnets netsets.IPNet) bool {
	for _, s := range ipnets.StringSlice() {
		if s == "0.0.0.0/0" {
			return true
		}
	}
	return false
}

// GetLoadBalancerSourceRanges verifies and parses the AnnotationLoadBalancerSourceRangesKey annotation from a service,
// extracting the source ranges to allow, and if not present returns a default (allow-all) value.
func GetLoadBalancerSourceRanges(annotations map[string]string) (netsets.IPNet, error) {
	val := annotations[AnnotationLoadBalancerSourceRangesKey]
	val = strings.TrimSpace(val)
	if val == "" {
		val = defaultLoadBalancerSourceRanges
	}
	specs := strings.Split(val, ",")
	ipnets, err := netsets.ParseIPNets(specs...)
	if err != nil {
		return nil, fmt.Errorf("Service annotation %s:%s is not valid. Expecting a comma-separated list of source IP ranges. For example, 10.0.0.0/24,192.168.2.0/24", AnnotationLoadBalancerSourceRangesKey, val)
	}
	return ipnets, nil
}
