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

package conntrack

import (
	"fmt"
	"strconv"
	"strings"

	"k8s.io/api/core/v1"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	"k8s.io/utils/exec"
)

// Utilities for dealing with conntrack

// NoConnectionToDelete is the error string returned by conntrack when no matching connections are found
const NoConnectionToDelete = "0 flow entries have been deleted"

func protoStr(proto v1.Protocol) string {
	return strings.ToLower(string(proto))
}

func parametersWithFamily(isIPv6 bool, parameters ...string) []string {
	if isIPv6 {
		parameters = append(parameters, "-f", "ipv6")
	}
	return parameters
}

// ClearEntriesForIP uses the conntrack tool to delete the conntrack entries
// for the UDP connections specified by the given service IP
func ClearEntriesForIP(execer exec.Interface, ip string, protocol v1.Protocol) error {
	parameters := parametersWithFamily(utilnet.IsIPv6String(ip), "-D", "--orig-dst", ip, "-p", protoStr(protocol))
	err := Exec(execer, parameters...)
	if err != nil && !strings.Contains(err.Error(), NoConnectionToDelete) {
		// TODO: Better handling for deletion failure. When failure occur, stale udp connection may not get flushed.
		// These stale udp connection will keep black hole traffic. Making this a best effort operation for now, since it
		// is expensive to baby-sit all udp connections to kubernetes services.
		return fmt.Errorf("error deleting connection tracking state for UDP service IP: %s, error: %v", ip, err)
	}
	return nil
}

// Exec executes the conntrack tool using the given parameters
func Exec(execer exec.Interface, parameters ...string) error {
	conntrackPath, err := execer.LookPath("conntrack")
	if err != nil {
		return fmt.Errorf("error looking for path of conntrack: %v", err)
	}
	output, err := execer.Command(conntrackPath, parameters...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("conntrack command returned: %q, error message: %s", string(output), err)
	}
	return nil
}

// Exists returns true if conntrack binary is installed.
func Exists(execer exec.Interface) bool {
	_, err := execer.LookPath("conntrack")
	return err == nil
}

// ClearEntriesForPort uses the conntrack tool to delete the conntrack entries
// for connections specified by the port.
// When a packet arrives, it will not go through NAT table again, because it is not "the first" packet.
// The solution is clearing the conntrack. Known issues:
// https://github.com/docker/docker/issues/8795
// https://github.com/kubernetes/kubernetes/issues/31983
func ClearEntriesForPort(execer exec.Interface, port int, isIPv6 bool, protocol v1.Protocol) error {
	if port <= 0 {
		return fmt.Errorf("Wrong port number. The port number must be greater than zero")
	}
	parameters := parametersWithFamily(isIPv6, "-D", "-p", protoStr(protocol), "--dport", strconv.Itoa(port))
	err := Exec(execer, parameters...)
	if err != nil && !strings.Contains(err.Error(), NoConnectionToDelete) {
		return fmt.Errorf("error deleting conntrack entries for UDP port: %d, error: %v", port, err)
	}
	return nil
}

// ClearEntriesForNAT uses the conntrack tool to delete the conntrack entries
// for connections specified by the {origin, dest} IP pair.
func ClearEntriesForNAT(execer exec.Interface, origin, dest string, protocol v1.Protocol) error {
	parameters := parametersWithFamily(utilnet.IsIPv6String(origin), "-D", "--orig-dst", origin, "--dst-nat", dest,
		"-p", protoStr(protocol))
	err := Exec(execer, parameters...)
	if err != nil && !strings.Contains(err.Error(), NoConnectionToDelete) {
		// TODO: Better handling for deletion failure. When failure occur, stale udp connection may not get flushed.
		// These stale udp connection will keep black hole traffic. Making this a best effort operation for now, since it
		// is expensive to baby sit all udp connections to kubernetes services.
		return fmt.Errorf("error deleting conntrack entries for UDP peer {%s, %s}, error: %v", origin, dest, err)
	}
	return nil
}
