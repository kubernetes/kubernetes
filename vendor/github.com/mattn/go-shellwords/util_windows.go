// +build windows,go1.6

package shellwords

import (
	"errors"
	"os"
	"os/exec"
	"strings"
)

func shellRun(line string) (string, error) {
	shell := os.Getenv("COMSPEC")
	b, err := exec.Command(shell, "/c", line).Output()
	if err != nil {
		if eerr, ok := err.(*exec.ExitError); ok {
			b = eerr.Stderr
		}
		return "", errors.New(err.Error() + ":" + string(b))
	}
	return strings.TrimSpace(string(b)), nil
}
