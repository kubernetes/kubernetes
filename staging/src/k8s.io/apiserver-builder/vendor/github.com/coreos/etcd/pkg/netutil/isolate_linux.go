// Copyright 2015 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package netutil

import (
	"fmt"
	"os/exec"
)

// DropPort drops all tcp packets that are received from the given port and sent to the given port.
func DropPort(port int) error {
	cmdStr := fmt.Sprintf("sudo iptables -A OUTPUT -p tcp --destination-port %d -j DROP", port)
	if _, err := exec.Command("/bin/sh", "-c", cmdStr).Output(); err != nil {
		return err
	}
	cmdStr = fmt.Sprintf("sudo iptables -A INPUT -p tcp --destination-port %d -j DROP", port)
	_, err := exec.Command("/bin/sh", "-c", cmdStr).Output()
	return err
}

// RecoverPort stops dropping tcp packets at given port.
func RecoverPort(port int) error {
	cmdStr := fmt.Sprintf("sudo iptables -D OUTPUT -p tcp --destination-port %d -j DROP", port)
	if _, err := exec.Command("/bin/sh", "-c", cmdStr).Output(); err != nil {
		return err
	}
	cmdStr = fmt.Sprintf("sudo iptables -D INPUT -p tcp --destination-port %d -j DROP", port)
	_, err := exec.Command("/bin/sh", "-c", cmdStr).Output()
	return err
}

// SetLatency adds latency in millisecond scale with random variations.
func SetLatency(ms, rv int) error {
	if rv > ms {
		rv = 1
	}
	cmdStr := fmt.Sprintf("sudo tc qdisc add dev eth0 root netem delay %dms %dms distribution normal", ms, rv)
	_, err := exec.Command("/bin/sh", "-c", cmdStr).Output()
	if err != nil {
		// the rule has already been added. Overwrite it.
		cmdStr = fmt.Sprintf("sudo tc qdisc change dev eth0 root netem delay %dms %dms distribution normal", ms, rv)
		_, err = exec.Command("/bin/sh", "-c", cmdStr).Output()
		if err != nil {
			return err
		}
	}
	return nil
}

// RemoveLatency resets latency configurations.
func RemoveLatency() error {
	_, err := exec.Command("/bin/sh", "-c", "sudo tc qdisc del dev eth0 root netem").Output()
	return err
}
