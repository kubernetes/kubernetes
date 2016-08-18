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

package kubeadmutil

import (
	"fmt"
	"net"
)

func GetDefaultHostIP() (string, error) {
	ifaces, err := net.Interfaces()
	if err != nil {
		return "", err
	}
	for _, i := range ifaces {
		addrs, err := i.Addrs()
		if err != nil {
			return "", err
		}
		for _, a := range addrs {
			if addr, ok := a.(*net.IPNet); ok && !addr.IP.IsLoopback() {
				if addr.IP.To4() != nil {
					return addr.IP.String(), nil
				}
			}
		}
	}
	return "", fmt.Errorf("Unable to autodetect IP address, please specify with --listen-ip")
}
