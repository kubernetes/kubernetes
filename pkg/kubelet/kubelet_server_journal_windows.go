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

// getLoggingCmd returns the powershell cmd and arguments for the given nodeLogQuery and boot
func getLoggingCmd(n *nodeLogQuery, services []string) (string, []string, error) {
	args := []string{
		"-NonInteractive",
		"-ExecutionPolicy", "Bypass",
		"-Command",
	}

	psCmd := "Get-WinEvent -FilterHashtable @{LogName='Application'"
	if n.SinceTime != nil {
		psCmd += fmt.Sprintf("; StartTime='%s'", n.SinceTime.Format(dateLayout))
	}
	if n.UntilTime != nil {
		psCmd += fmt.Sprintf("; EndTime='%s'", n.UntilTime.Format(dateLayout))
	}
	var providers []string
	for _, service := range services {
		if len(service) > 0 {
			providers = append(providers, "'"+service+"'")
		}
	}
	if len(providers) > 0 {
		psCmd += fmt.Sprintf("; ProviderName=%s", strings.Join(providers, ","))
	}
	psCmd += "}"
	if n.TailLines != nil {
		psCmd += fmt.Sprintf(" -MaxEvents %d", *n.TailLines)
	}
	psCmd += " | Sort-Object TimeCreated"
	if len(n.Pattern) > 0 {
		psCmd += fmt.Sprintf(" | Where-Object -Property Message -Match '%s'", n.Pattern)
	}
	psCmd += " | Format-Table -AutoSize -Wrap"

	args = append(args, psCmd)

	return powershellExe, args, nil
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
