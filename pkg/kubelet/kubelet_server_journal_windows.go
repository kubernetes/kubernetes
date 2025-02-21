//go:build windows

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

const powershellExe = "PowerShell.exe"

// getLoggingCmd returns the powershell cmd, arguments, and environment variables for the given nodeLogQuery and boot.
// All string inputs are environment variables to stop subcommands expressions from being executed.
// The return values are:
// - cmd: the command to be executed
// - args: arguments to the command
// - cmdEnv: environment variables when the command will be executed
func getLoggingCmd(n *nodeLogQuery, services []string) (cmd string, args []string, cmdEnv []string, err error) {
	cmdEnv = getLoggingCmdEnv(n, services)

	var includeSinceTime, includeUntilTime, includeTailLines, includePattern bool
	if n.SinceTime != nil {
		includeSinceTime = true
	}
	if n.UntilTime != nil {
		includeUntilTime = true
	}
	if n.TailLines != nil {
		includeTailLines = true
	}
	if len(n.Pattern) > 0 {
		includePattern = true
	}

	var includeServices []bool
	for _, service := range services {
		includeServices = append(includeServices, len(service) > 0)
	}

	args = getLoggingCmdArgs(includeSinceTime, includeUntilTime, includeTailLines, includePattern, includeServices)

	return powershellExe, args, cmdEnv, nil
}

// getLoggingCmdArgs returns arguments that need to be passed to powershellExe
func getLoggingCmdArgs(includeSinceTime, includeUntilTime, includeTailLines, includePattern bool, services []bool) (args []string) {
	args = []string{
		"-NonInteractive",
		"-ExecutionPolicy", "Bypass",
		"-Command",
	}

	psCmd := `Get-WinEvent -FilterHashtable @{LogName='Application'`

	if includeSinceTime {
		psCmd += fmt.Sprintf(`; StartTime="$Env:kubelet_sinceTime"`)
	}
	if includeUntilTime {
		psCmd += fmt.Sprintf(`; EndTime="$Env:kubelet_untilTime"`)
	}

	var providers []string
	for i := range services {
		if services[i] {
			providers = append(providers, fmt.Sprintf("$Env:kubelet_provider%d", i))
		}
	}

	if len(providers) > 0 {
		psCmd += fmt.Sprintf("; ProviderName=%s", strings.Join(providers, ","))
	}

	psCmd += `}`
	if includeTailLines {
		psCmd += fmt.Sprint(` -MaxEvents $Env:kubelet_tailLines`)
	}
	psCmd += ` | Sort-Object TimeCreated`

	if includePattern {
		psCmd += fmt.Sprintf(` | Where-Object -Property Message -Match "$Env:kubelet_pattern"`)
	}
	psCmd += ` | Format-Table -AutoSize -Wrap`

	args = append(args, psCmd)

	return args
}

// getLoggingCmdEnv returns the environment variables that will be present when powershellExe is executed
func getLoggingCmdEnv(n *nodeLogQuery, services []string) (cmdEnv []string) {
	if n.SinceTime != nil {
		cmdEnv = append(cmdEnv, fmt.Sprintf("kubelet_sinceTime=%s", n.SinceTime.Format(dateLayout)))
	}
	if n.UntilTime != nil {
		cmdEnv = append(cmdEnv, fmt.Sprintf("kubelet_untilTime=%s", n.UntilTime.Format(dateLayout)))
	}

	for i, service := range services {
		if len(service) > 0 {
			cmdEnv = append(cmdEnv, fmt.Sprintf("kubelet_provider%d=%s", i, service))
		}
	}

	if n.TailLines != nil {
		cmdEnv = append(cmdEnv, fmt.Sprintf("kubelet_tailLines=%d", *n.TailLines))
	}

	if len(n.Pattern) > 0 {
		cmdEnv = append(cmdEnv, fmt.Sprintf("kubelet_pattern=%s", n.Pattern))
	}

	return cmdEnv
}

// checkForNativeLogger always returns true for Windows
func checkForNativeLogger(ctx context.Context, service string) bool {
	cmd := exec.CommandContext(ctx, powershellExe, []string{
		"-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command",
		fmt.Sprintf("Get-WinEvent -ListProvider %s | Format-Table -AutoSize", service)}...)

	_, err := cmd.CombinedOutput()
	if err != nil {
		// Get-WinEvent will return ExitError if the service is not listed as a provider
		if _, ok := err.(*exec.ExitError); ok {
			return false
		}
		// Other errors imply that CombinedOutput failed before the command was executed,
		// so lets to get the logs using Get-WinEvent at the call site instead of assuming
		// the service is logging to a file
	}
	return true
}
