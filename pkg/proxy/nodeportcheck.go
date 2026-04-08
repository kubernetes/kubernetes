/*
Copyright 2026 The Kubernetes Authors.

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

package proxy

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"k8s.io/klog/v2"
)

const ephemeralPortRangeFile = "/proc/sys/net/ipv4/ip_local_port_range"

// WarnIfNodePortsOverlapEphemeralRange reads the node's ephemeral port range from
// /proc/sys/net/ipv4/ip_local_port_range and logs a warning for any NodePort or
// HealthCheckNodePort in svcPortMap that falls within that range. Such overlap can
// cause connection failures when the kernel assigns the same port for outbound traffic.
func WarnIfNodePortsOverlapEphemeralRange(logger klog.Logger, svcPortMap ServicePortMap) {
	low, high, err := readEphemeralPortRange()
	if err != nil {
		logger.Error(err, "Failed to read ephemeral port range, skipping overlap check")
		return
	}
	warnNodePortOverlaps(logger, svcPortMap, low, high)
}

func warnNodePortOverlaps(logger klog.Logger, svcPortMap ServicePortMap, low, high int) {
	for svcName, svc := range svcPortMap {
		if np := svc.NodePort(); np != 0 && np >= low && np <= high {
			logger.V(0).Info("NodePort overlaps with ephemeral port range and may conflict with outbound connections",
				"service", svcName, "nodePort", np, "ephemeralRange", fmt.Sprintf("%d-%d", low, high))
		}
		if hcnp := svc.HealthCheckNodePort(); hcnp != 0 && hcnp >= low && hcnp <= high {
			logger.V(0).Info("HealthCheckNodePort overlaps with ephemeral port range and may conflict with outbound connections",
				"service", svcName, "healthCheckNodePort", hcnp, "ephemeralRange", fmt.Sprintf("%d-%d", low, high))
		}
	}
}

func readEphemeralPortRange() (int, int, error) {
	data, err := os.ReadFile(ephemeralPortRangeFile)
	if err != nil {
		return 0, 0, err
	}
	return parseEphemeralPortRange(string(data))
}

func parseEphemeralPortRange(data string) (int, int, error) {
	fields := strings.Fields(strings.TrimSpace(data))
	if len(fields) != 2 {
		return 0, 0, fmt.Errorf("unexpected format in %s: %q", ephemeralPortRangeFile, data)
	}
	low, err := strconv.Atoi(fields[0])
	if err != nil {
		return 0, 0, fmt.Errorf("failed to parse low value in %s: %w", ephemeralPortRangeFile, err)
	}
	high, err := strconv.Atoi(fields[1])
	if err != nil {
		return 0, 0, fmt.Errorf("failed to parse high value in %s: %w", ephemeralPortRangeFile, err)
	}
	return low, high, nil
}
