// +build !go1.6

package shellwords

import (
	"os"
	"os/exec"
	"runtime"
	"strings"
)

func shellRun(line string) (string, error) {
	var b []byte
	var err error
	if runtime.GOOS == "windows" {
		b, err = exec.Command(os.Getenv("COMSPEC"), "/c", line).Output()
	} else {
		b, err = exec.Command(os.Getenv("SHELL"), "-c", line).Output()
	}
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(b)), nil
}
