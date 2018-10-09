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

package net

import (
	"net"
)

// IsIPv6 returns if netIP is IPv6.
func IsIPv6(netIP net.IP) bool {
	return netIP != nil && netIP.To4() == nil
}

// IsIPv6String returns if ip is IPv6.
func IsIPv6String(ip string) bool {
	netIP := net.ParseIP(ip)
	return IsIPv6(netIP)
}

// IsIPv6CIDR returns if cidr is IPv6.
// This assumes cidr is a valid CIDR.
func IsIPv6CIDR(cidr string) bool {
	ip, _, _ := net.ParseCIDR(cidr)
	return IsIPv6(ip)
}

// FilterIncorrectIPVersion filters out the incorrect IP version case from a slice of IP strings.
func FilterIncorrectIPVersion(ipStrings []string, isIPv6Mode bool) ([]string, []string) {
	return filterWithCondition(ipStrings, isIPv6Mode, IsIPv6String)
}

// FilterIncorrectCIDRVersion filters out the incorrect IP version case from a slice of CIDR strings.
func FilterIncorrectCIDRVersion(ipStrings []string, isIPv6Mode bool) ([]string, []string) {
	return filterWithCondition(ipStrings, isIPv6Mode, IsIPv6CIDR)
}

func filterWithCondition(strs []string, expectedCondition bool, conditionFunc func(string) bool) ([]string, []string) {
	var corrects, incorrects []string
	for _, str := range strs {
		if conditionFunc(str) != expectedCondition {
			incorrects = append(incorrects, str)
		} else {
			corrects = append(corrects, str)
		}
	}
	return corrects, incorrects
}
