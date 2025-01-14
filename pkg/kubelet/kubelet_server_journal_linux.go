//go:build linux

/*
Copyright 2022 The Kubernetes Authors.

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

package kubelet

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
)

// getLoggingCmd returns the journalctl cmd and arguments for the given nodeLogQuery and boot. Note that
// services are explicitly passed here to account for the heuristics.
// The return values are:
// - cmd: the command to be executed
// - args: arguments to the command
// - cmdEnv: environment variables when the command will be executed
func getLoggingCmd(n *nodeLogQuery, services []string) (cmd string, args []string, cmdEnv []string, err error) {
	args = []string{
		"--utc",
		"--no-pager",
		"--output=short-precise",
	}
	if n.SinceTime != nil {
		args = append(args, fmt.Sprintf("--since=%s", n.SinceTime.Format(dateLayout)))
	}
	if n.UntilTime != nil {
		args = append(args, fmt.Sprintf("--until=%s", n.SinceTime.Format(dateLayout)))
	}
	if n.TailLines != nil {
		args = append(args, "--pager-end", fmt.Sprintf("--lines=%d", *n.TailLines))
	}
	for _, service := range services {
		if len(service) > 0 {
			args = append(args, "--unit="+service)
		}
	}
	if len(n.Pattern) > 0 {
		args = append(args, "--grep="+n.Pattern)
	}

	if n.Boot != nil {
		args = append(args, "--boot", fmt.Sprintf("%d", *n.Boot))
	}

	return "journalctl", args, nil, nil
}

// checkForNativeLogger checks journalctl output for a service
func checkForNativeLogger(ctx context.Context, service string) bool {
	// This will return all the journald units
	cmd := exec.CommandContext(ctx, "journalctl", []string{"--field", "_SYSTEMD_UNIT"}...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		// Returning false to allow checking if the service is logging to a file
		return false
	}

	// journalctl won't return an error if we try to fetch logs for a non-existent service,
	// hence we search for it in the list of services known to journalctl
	return strings.Contains(string(output), service+".service")
}
