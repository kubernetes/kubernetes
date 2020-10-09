// +build !windows

/*
Copyright 2019 The Kubernetes Authors.

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

package dns

import (
	"strings"
)

const etcHostsFile = "/etc/hosts"

// A /etc/resolv.conf file managed by kubelet looks like this:
// nameserver DNS_CLUSTER_IP
// search test-dns.svc.cluster.local svc.cluster.local cluster.local q53aahaikqaehcai3ylfqdtc5b.bx.internal.cloudapp.net
// options ndots:5

// GetDNSSuffixList reads DNS config file and returns the list of configured DNS suffixes
func GetDNSSuffixList() []string {
	fileData := readFile("/etc/resolv.conf")
	lines := strings.Split(fileData, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "search") {
			// omit the starting "search".
			return strings.Split(line, " ")[1:]
		}
	}

	panic("Could not find DNS search list!")
}

func getDNSServerList() []string {
	fileData := readFile("/etc/resolv.conf")
	lines := strings.Split(fileData, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "nameserver") {
			// omit the starting "nameserver".
			return strings.Split(line, " ")[1:]
		}
	}

	panic("Could not find DNS search list!")
}
