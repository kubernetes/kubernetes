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

package cloudprovider

import (
	"fmt"
	"net"
	"strings"
)

const (
	defaultLBSourceRange = "0.0.0.0/0"
)

type IPNetSet map[string]*net.IPNet

func ParseIPNetSet(specs []string) (IPNetSet, error) {
	ipnetset := make(IPNetSet)
	for _, spec := range specs {
		spec = strings.TrimSpace(spec)
		_, ipnet, err := net.ParseCIDR(spec)
		if err != nil {
			return nil, err
		}
		k := ipnet.String() // In case of normalization
		ipnetset[k] = ipnet
	}
	return ipnetset, nil
}

// StringSlice returns a []string with the String representation of each element in the set.
// Order is undefined.
func (s IPNetSet) StringSlice() []string {
	a := make([]string, 0, len(s))
	for k := range s {
		a = append(a, k)
	}
	return a
}

// Equal checks if two IPNetSets are equal (ignoring order)
func (l IPNetSet) Equal(r IPNetSet) bool {
	if len(l) != len(r) {
		return false
	}

	for k := range l {
		_, found := r[k]
		if !found {
			return false
		}
	}
	return true
}

// GetSourceRangeAnnotations verifies and parses the LBAnnotationAllowSourceRange annotation from a service,
// extracting the source ranges to allow, and if not present returns a default (allow-all) value.
func GetSourceRangeAnnotations(annotation map[string]string) (IPNetSet, error) {
	val := annotation[LBAnnotationAllowSourceRange]
	val = strings.TrimSpace(val)
	if val == "" {
		val = defaultLBSourceRange
	}
	specs := strings.Split(val, ",")
	ipnets, err := ParseIPNetSet(specs)
	if err != nil {
		return nil, fmt.Errorf("Service annotation %s:%s is not valid. Expecting a comma-separated list of source IP ranges. For example, 10.0.0.0/24,192.168.2.0/24", LBAnnotationAllowSourceRange, val)
	}
	return ipnets, nil
}
