//go:build windows
// +build windows

package kubelet

import (
	"fmt"
	"strings"
)

// getLoggingCmd returns the powershell cmd and arguments for the given journalArgs and boot
func getLoggingCmd(a *journalArgs, boot int) (string, []string) {
	// The WinEvent log does not support querying by boot
	// Set the cmd to return true on windows in case boot is not 0
	if boot != 0 {
		return "cd.", []string{}
	}

	args := []string{
		"-NonInteractive",
		"-ExecutionPolicy", "Bypass",
		"-Command",
	}

	psCmd := "Get-WinEvent -FilterHashtable @{LogName='Application'"
	if len(a.Since) > 0 {
		psCmd += fmt.Sprintf("; StartTime='%s'", a.Since)
	}
	if len(a.Until) > 0 {
		psCmd += fmt.Sprintf("; EndTime='%s'", a.Until)
	}
	var units []string
	for _, unit := range a.Units {
		if len(unit) > 0 {
			units = append(units, "'"+unit+"'")
		}
	}
	if len(units) > 0 {
		psCmd += fmt.Sprintf("; ProviderName=%s", strings.Join(units, ","))
	}
	psCmd += "}"
	if a.Tail > 0 {
		psCmd += fmt.Sprintf(" -MaxEvents %d", a.Tail)
	}
	psCmd += " | Sort-Object TimeCreated"
	if len(a.Pattern) > 0 {
		psCmd += fmt.Sprintf(" | Where-Object -Property Message -Match %s", a.Pattern)
	}
	psCmd += " | Format-Table -AutoSize -Wrap"

	args = append(args, psCmd)

	return "PowerShell.exe", args
}
