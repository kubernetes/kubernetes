// +build windows

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

package dockershim

import (
	"fmt"
	"io"
	"net"
	"strings"

	"github.com/Microsoft/hcsshim"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
	"k8s.io/kubernetes/pkg/util/netsh"
	utilexec "k8s.io/utils/exec"
)

func portForward(client libdocker.Interface, podSandboxID string, port int32, stream io.ReadWriteCloser) error {
	endpoints, err := hcsshim.HNSListEndpointRequest()
	if err != nil {
		return err
	}

	var match *hcsshim.HNSEndpoint
	for _, endpoint := range endpoints {
		if strings.HasPrefix(endpoint.Name, podSandboxID) {
			match = &endpoint
			break
		}
	}
	if match == nil {
		return fmt.Errorf("couldn't find a matching endpoint for podSandboxID %s", podSandboxID)
	}
	netshClient := netsh.New(utilexec.New())
	randomPort, err := findOpenPort()
	if err != nil {
		return err
	}

	netshAddArgs := []string{
		"interface",
		"portproxy",
		"add",
		"v4tov4",
		fmt.Sprintf("listenport=%d", randomPort),
		fmt.Sprintf("connectaddress=%s", match.IPAddress.String()),
		fmt.Sprintf("connectport=%d", port)}

	ok, err := netshClient.EnsurePortProxyRule(netshAddArgs)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("error adding netsh rule: %s", netshAddArgs)
	}
	conn, err := net.Dial("tcp", fmt.Sprintf("localhost:%d", randomPort))
	if err != nil {
		return err
	}
	defer conn.Close()

	go func() {
		io.Copy(conn, stream)
	}()
	io.Copy(stream, conn)

	netshDelArgs := []string{
		"interface",
		"portproxy",
		"delete",
		"v4tov4",
		fmt.Sprintf("listenport=%d", randomPort)}
	return netshClient.DeletePortProxyRule(netshDelArgs)
}

func findOpenPort() (int, error) {
	addr, err := net.ResolveTCPAddr("tcp", "localhost:0")
	if err != nil {
		return 0, err
	}
	l, err := net.ListenTCP("tcp", addr)
	if err != nil {
		return 0, err
	}
	defer l.Close()

	return l.Addr().(*net.TCPAddr).Port, nil
}
