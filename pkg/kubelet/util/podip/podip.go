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

package podip

import (
	"bytes"
	"fmt"
	"net"
	"strings"

	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/util/ioutils"
)

func getIpGetterCommand(interfaceName, addrType string) []string {
	return []string{"ip", "-o", addrType, "addr", "show", "dev", interfaceName, "scope", "global"}

}

func parseIP(output string) (net.IP, error) {
	lines := strings.Split(string(output), "\n")
	if len(lines) < 1 {
		return nil, fmt.Errorf("Unexpected command output %s", output)
	}
	fields := strings.Fields(lines[0])
	if len(fields) < 4 {
		return nil, fmt.Errorf("Unexpected address output %s ", lines[0])
	}
	ip, _, err := net.ParseCIDR(fields[3])
	if err != nil {
		return nil, fmt.Errorf("CNI failed to parse ip from output %s due to %v", output, err)
	}

	return ip, nil
}

func GetPodIP(r kubecontainer.ContainerCommandRunner, containerID kubecontainer.ContainerID, interfaceName string) (net.IP, error) {
	var buffer bytes.Buffer
	output := ioutils.WriteCloserWrapper(&buffer)
	err := r.ExecInContainer(containerID, getIpGetterCommand(interfaceName, "-4"), nil, output, output, false, nil)
	if err != nil {
		return nil, err
	}
	ip, err := parseIP(buffer.String())
	if err != nil {
		// Fall back to IPv6 address if no IPv4 address is present
		err = r.ExecInContainer(containerID, getIpGetterCommand(interfaceName, "-6"), nil, output, output, false, nil)
	}
	if err != nil {
		return nil, err
	}
	ip, err = parseIP(buffer.String())
	if err != nil {
		return nil, err
	}
	return ip, nil

}
