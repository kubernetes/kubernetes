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

package util

import (
	"fmt"
	"strings"

	"k8s.io/kubernetes/pkg/util/exec"

	"github.com/golang/glog"
)

// Utilities for dealing with conntrack

const noConnectionToDelete = "0 flow entries have been deleted"

// DeleteServiceConnection uses the conntrack tool to delete the conntrack entries
// for the UDP connections specified by the given service IPs
func DeleteServiceConnections(execer exec.Interface, svcIPs []string) {
	for _, ip := range svcIPs {
		glog.V(2).Infof("Deleting connection tracking state for service IP %s", ip)
		err := ExecConntrackTool(execer, "-D", "--orig-dst", ip, "-p", "udp")
		if err != nil && !strings.Contains(err.Error(), noConnectionToDelete) {
			// TODO: Better handling for deletion failure. When failure occur, stale udp connection may not get flushed.
			// These stale udp connection will keep black hole traffic. Making this a best effort operation for now, since it
			// is expensive to baby-sit all udp connections to kubernetes services.
			glog.Errorf("conntrack returned error: %v", err)
		}
	}
}

// ExecConntrackTool executes the conntrack tool using the given parameters
func ExecConntrackTool(execer exec.Interface, parameters ...string) error {
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
