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
	"io"
	"net"
	"testing"

	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/util/term"
)

func TestParseIpV4(t *testing.T) {
	output := "2: eth0    inet 10.10.0.2/24 brd 10.10.0.255 scope global eth0\\       valid_lft forever preferred_lft forever"
	expected, _, _ := net.ParseCIDR("10.10.0.2/24")
	ip, _ := parseIP(output)
	if string(expected) != string(ip) {
		t.Errorf("Result is not expected: %v != %v", ip, expected)
	}
}

func TestParseIpV6(t *testing.T) {
	output := "2: eth0    inet6 2001:db8:0:f101::1/64 scope global \\       valid_lft forever preferred_lft forever"
	expected, _, _ := net.ParseCIDR("2001:db8:0:f101::1/64")
	ip, _ := parseIP(output)
	if string(expected) != string(ip) {
		t.Errorf("Result is not expected: %v != %v", ip, expected)
	}
}

type MockCommandRunner struct {
	toWrite []byte
}

func (r MockCommandRunner) ExecInContainer(containerID kubecontainer.ContainerID, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan term.Size) error {
	stdout.Write(r.toWrite)
	return nil
}

func (r MockCommandRunner) PortForward(pod *kubecontainer.Pod, port uint16, stream io.ReadWriteCloser) error {
	return nil
}

func TestGetPodIpV4(t *testing.T) {
	mockCommander := MockCommandRunner{
		toWrite: []byte("2: eth0    inet 10.10.0.2/24 brd 10.10.0.255 scope global eth0\\       valid_lft forever preferred_lft forever"),
	}
	expected, _, _ := net.ParseCIDR("10.10.0.2/24")
	ip, _ := GetPodIP(mockCommander, kubecontainer.ContainerID{Type: "test", ID: "abc1234"}, "eth0")
	if string(ip) != string(expected) {
		t.Errorf("Result is not expected: %v != %v", ip, expected)
	}
}
