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

package sysctl

import (
	"fmt"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/util/version"
)

func Test_getSafeSysctlAllowlist(t *testing.T) {
	tests := []struct {
		name       string
		getVersion func() (*version.Version, error)
		want       []string
	}{
		{
			name: "failed to get kernelVersion, only return safeSysctls with no kernelVersion limit",
			getVersion: func() (*version.Version, error) {
				return nil, fmt.Errorf("fork error")
			},
			want: []string{
				"kernel.shm_rmid_forced",
				"net.ipv4.ip_local_port_range",
				"net.ipv4.tcp_syncookies",
				"net.ipv4.ping_group_range",
				"net.ipv4.ip_unprivileged_port_start",
				"net.ipv4.tcp_rmem",
				"net.ipv4.tcp_wmem",
			},
		},
		{
			name: "kernelVersion is 3.18.0, return safeSysctls with no kernelVersion limit and net.ipv4.ip_local_reserved_ports",
			getVersion: func() (*version.Version, error) {
				kernelVersionStr := "3.18.0-957.27.2.el7.x86_64"
				return version.ParseGeneric(kernelVersionStr)
			},
			want: []string{
				"kernel.shm_rmid_forced",
				"net.ipv4.ip_local_port_range",
				"net.ipv4.tcp_syncookies",
				"net.ipv4.ping_group_range",
				"net.ipv4.ip_unprivileged_port_start",
				"net.ipv4.ip_local_reserved_ports",
				"net.ipv4.tcp_rmem",
				"net.ipv4.tcp_wmem",
			},
		},
		{
			name: "kernelVersion is 5.15.0, return safeSysctls with no kernelVersion limit and kernelVersion below 5.15.0",
			getVersion: func() (*version.Version, error) {
				kernelVersionStr := "5.15.0-75-generic"
				return version.ParseGeneric(kernelVersionStr)
			},
			want: []string{
				"kernel.shm_rmid_forced",
				"net.ipv4.ip_local_port_range",
				"net.ipv4.tcp_syncookies",
				"net.ipv4.ping_group_range",
				"net.ipv4.ip_unprivileged_port_start",
				"net.ipv4.ip_local_reserved_ports",
				"net.ipv4.tcp_keepalive_time",
				"net.ipv4.tcp_fin_timeout",
				"net.ipv4.tcp_keepalive_intvl",
				"net.ipv4.tcp_keepalive_probes",
				"net.ipv4.tcp_rmem",
				"net.ipv4.tcp_wmem",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := getSafeSysctlAllowlist(tt.getVersion); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("getSafeSysctlAllowlist() = %v, want %v", got, tt.want)
			}
		})
	}
}
