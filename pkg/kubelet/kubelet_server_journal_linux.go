//go:build linux
// +build linux

package kubelet

import (
	"fmt"
)

// getLoggingCmd returns the journalctl cmd and arguments for the given journalArgs and boot
func getLoggingCmd(a *journalArgs, boot int) (string, []string) {
	args := []string{
		"--utc",
		"--no-pager",
	}
	if len(a.Since) > 0 {
		args = append(args, "--since="+a.Since)
	}
	if len(a.Until) > 0 {
		args = append(args, "--until="+a.Until)
	}
	if a.Tail > 0 {
		args = append(args, "--pager-end", fmt.Sprintf("--lines=%d", a.Tail))
	}
	if len(a.Format) > 0 {
		args = append(args, "--output="+a.Format)
	}
	for _, unit := range a.Units {
		if len(unit) > 0 {
			args = append(args, "--unit="+unit)
		}
	}
	if len(a.Pattern) > 0 {
		args = append(args, "--grep="+a.Pattern)
		args = append(args, fmt.Sprintf("--case-sensitive=%t", a.CaseSensitive))
	}

	args = append(args, "--boot", fmt.Sprintf("%d", boot))

	return "journalctl", args
}
